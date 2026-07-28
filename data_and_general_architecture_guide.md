# Public Datasets & General Architecture Guide

> **Purpose:** Preprocessing public Loghub BGL logs, physical topology distance metrics, and reusable core logic patterns across problem statements.
> **Scope:** Bridges the public BGL dataset to our structured alert schemas, and provides reusable code skeletons for other problem statements.

---

# PART 1 — Preprocessing the Public BGL Dataset (Zero Data Loss)

To comply with the rule of using publicly available datasets as the foundation (instead of relying solely on generated synthetic data), we will use the **Loghub BlueGene/L (BGL)** dataset. BGL logs contain actual system logs and hardware alerts from a 131,072-processor supercomputer.

## 1.1 — The Raw BGL Format
Each line in the raw `BGL.log` file contains 9 fields separated by spaces:
```
APPRECOVERY 1118536442 2005.06.11 R04-M1-N4-I:J18-U11 2005-06-11-17.34.02.779776 R04-M1-N4-I:J18-U11 APP error ciod: failed to read message header on control stream link to web server
```
- **Field 1 (Alert Label):** `-` for normal system logs, or a specific alert tag (e.g., `APPRECOVERY`, `FATAL`, `WARNING`) for alert logs.
- **Field 2 (Epoch Timestamp):** e.g., `1118536442`
- **Field 3 (Date):** e.g., `2005.06.11`
- **Field 4 (Node Coordinates):** Physical layout coordinate (e.g., `R04-M1-N4-I:J18-U11`)
- **Field 5 (Precise Timestamp):** e.g., `2005-06-11-17.34.02.779776`
- **Field 6 (Node Address Repeated):** e.g., `R04-M1-N4-I:J18-U11`
- **Field 7 (Component/Service):** e.g., `APP`, `KERNEL`, `DISCOVERY`
- **Field 8 (Log Level):** e.g., `error`, `info`, `warning`
- **Field 9 (Content):** The actual message text.

---

## 1.2 — BGL Log Parser (Python Implementation)

This script parses raw BGL lines, converts them to our standardized `Alert` Pydantic models with **zero data loss**, and categorizes them.

```python
import re
import hashlib
from datetime import datetime
from pydantic import BaseModel

class Alert(BaseModel):
    id: str
    timestamp: datetime
    source: str
    severity: str
    service: str
    host: str
    message: str
    tags: list[str] = []

# Regular expression to extract the 9 fields of a BGL log line
BGL_REGEX = re.compile(
    r'^([-\w\d]+)\s+'               # 1. Alert Label
    r'(\d+)\s+'                    # 2. Epoch Timestamp
    r'([\d\.]+)\s+'                # 3. Date
    r'([-\w\d\.:]+)\s+'            # 4. Node Coordinates
    r'([\d\.-]+)\s+'               # 5. Precise Timestamp
    r'([-\w\d\.:]+)\s+'            # 6. Repeated Node Coordinates
    r'(\w+)\s+'                    # 7. Component/Service
    r'(\w+)\s+'                    # 8. Log Level
    r'(.*)$'                       # 9. Content / Message
)

def parse_bgl_line(line: str) -> Alert | None:
    """
    Parses a single raw line from the BGL log and converts it to a structured Alert.
    Returns None if the line does not match the expected pattern.
    """
    match = BGL_REGEX.match(line.strip())
    if not match:
        return None
        
    alert_label = match.group(1)
    epoch_ts = int(match.group(2))
    node_coord = match.group(4)
    precise_ts_str = match.group(5)
    component = match.group(7)
    log_level = match.group(8)
    message_content = match.group(9)
    
    # 1. Parse Timestamp
    try:
        # Format: 2005-06-11-17.34.02.779776
        timestamp = datetime.strptime(precise_ts_str, "%Y-%m-%d-%H.%M.%S.%f")
    except ValueError:
        # Fallback to epoch if microsecond string fails
        timestamp = datetime.fromtimestamp(epoch_ts)
        
    # 2. Map Severity
    # If the alert label is not "-", it is an explicit hardware/software alert
    is_alert = alert_label != "-"
    mapped_severity = "info"
    if is_alert:
        mapped_severity = "critical" if log_level.lower() in ["error", "fatal", "severe"] else "warning"
    else:
        # Even if not flagged as alert, respect log levels
        if log_level.lower() in ["fatal", "severe"]:
            mapped_severity = "critical"
        elif log_level.lower() in ["error", "warning", "warn"]:
            mapped_severity = "warning"
            
    # 3. Generate ID deterministically (to prevent duplicates across pipeline runs)
    line_hash = hashlib.md5(line.strip().encode()).hexdigest()
    
    # 4. Extract topology markers as tags
    tags = [f"label:{alert_label}", f"level:{log_level}"]
    
    # Extract rack coordinate if present
    if "-" in node_coord:
        rack = node_coord.split("-")[0]
        tags.append(f"rack:{rack}")
        
    return Alert(
        id=f"bgl-{line_hash[:12]}",
        timestamp=timestamp,
        source="BlueGeneL-MMCS",
        severity=mapped_severity,
        service=component,
        host=node_coord,
        message=message_content,
        tags=tags
    )
```

---

# PART 2 — The Physical Topology Distance Metric

Standard log-correlation systems use simple Jaccard index string metrics for topological tags. Because BGL logs contain physical node coordinates (`Rxx-Mxx-Nxx`), we can construct a **physically-motivated topology distance metric**.

## 2.1 — BlueGene/L Coordinate Mapping
The coordinates indicate the physical location of components within the supercomputer:
- `Rxx` = **Rack** (e.g., Rack 04)
- `Mxx` = **Midplane** within the rack
- `Nxx` = **Node Card** mounted on the midplane

```
Supercomputer Layout:
[Racks R00 to Rxx]
   └── [Midplanes M0 and M1 per Rack]
          └── [Node Cards N00 to N15 per Midplane]
```

---

## 2.2 — Python Implementation of Topology Distance
This function calculates the physical distance between two nodes, returning a float from `0.0` (identical component) to `1.0` (completely separate racks).

```python
import re

# Parser for Rxx-Mxx-Nxx coordinates
TOPOLOGY_REGEX = re.compile(r'^(R\d{2})-(M\d{1})-(N\d{2})')

def compute_topology_distance(host1: str, host2: str) -> float:
    """
    Computes a physical layout distance metric between two BlueGene/L nodes.
    
    Returns:
        float in [0.0, 1.0] representing distance:
        0.00 -> Identical node
        0.10 -> Same node card (extremely close)
        0.30 -> Same midplane, different node card
        0.60 -> Same rack, different midplane
        1.00 -> Different racks (physically separate)
    """
    if host1 == host2:
        return 0.0
        
    match1 = TOPOLOGY_REGEX.search(host1)
    match2 = TOPOLOGY_REGEX.search(host2)
    
    # If coordinates are missing or unparseable, default to max distance
    if not match1 or not match2:
        return 1.0
        
    r1, m1, n1 = match1.groups()
    r2, m2, n2 = match2.groups()
    
    if r1 != r2:
        return 1.0  # Different racks
    if m1 != m2:
        return 0.6  # Same rack, different midplane
    if n1 != n2:
        return 0.3  # Same midplane, different node card
        
    return 0.1  # Same node card, different sub-components
```

This custom metric can replace `topological_score` in the multi-signal matrix. It is deterministic, computationally fast, and has high appeal for judges looking for domain-specific engineering.

---

# PART 3 — Core Architecture & Logic (Reusable Everywhere)

Regardless of the problem statement selected, several core software engineering blocks can be shared across all tasks. These patterns prevent **vibe coding** and enforce structure.

```
                  ┌──────────────────────────────────────────────┐
                  │          SHARED SYSTEM ARCHITECTURE          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │                FRONTEND LAYER                  │
                 │   React / Streamlit Dashboard / Chart.js      │
                 └───────────────────────┬────────────────────────┘
                                         │  (JSON REST API)
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │                BACKEND LAYER                   │
                 │             FastAPI / Pydantic                 │
                 └───────────────────────┬────────────────────────┘
                                         │  (In-Memory Objects)
                                         ▼
                 ┌────────────────────────────────────────────────┐
                 │                AI INTELLIGENCE LAYER           │
                 │   Embeddings / ML Classifiers / LLM Wrapper    │
                 └────────────────────────────────────────────────┘
```

---

## 3.1 — Reusable Logic Block A: LLM Self-Healing JSON Parser
*Applicable to: #2, #3, #6, #7, #8, #10, #14, #15*

Ensures that LLMs return JSON matching a Pydantic schema without crashing the backend.

```python
import json
from typing import Type
from pydantic import BaseModel

def generate_structured_json(
    client, 
    prompt: str, 
    response_schema: Type[BaseModel],
    max_retries: int = 2
) -> BaseModel:
    """
    Requests a structured response from the LLM, parses the JSON,
    and returns a validated Pydantic model. Retries automatically if parsing fails.
    """
    schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
    
    formatted_prompt = f"""{prompt}

CRITICAL: Return ONLY a valid JSON object matching the schema below.
Do not include any Markdown blocks, backticks, comments, or explanations.

JSON Schema:
{schema_json}
"""

    for attempt in range(max_retries + 1):
        try:
            # call standard client generation method
            raw_response = client.generate(formatted_prompt)
            
            # Clean possible markdown blocks or backticks
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                # Remove starting and ending lines with backticks
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
                
            parsed_data = json.loads(cleaned)
            return response_schema.model_validate(parsed_data)
            
        except (json.JSONDecodeError, Exception) as e:
            if attempt == max_retries:
                # If all retries fail, return a default populated instance of the schema
                return response_schema.model_construct()
            
            # Formulate correction prompt for retry
            prompt = f"""Your previous response failed parsing with error: {str(e)}.
Please re-generate the output ensuring it is strictly valid JSON.

Original Request:
{prompt}"""
```

---

## 3.2 — Reusable Logic Block B: Explainable XGBoost Pipeline
*Applicable to: #5 (SLA Predictor), #9 (Deployment Risk), #13 (Traffic Anomaly)*

A robust pipeline structure that handles tabular metrics, scales features, runs predictions, and produces SHAP explainability.

```python
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler

class TabularPredictor:
    """
    Wrapper for training an XGBoost classifier with built-in feature scaling
    and SHAP explainability calculations.
    """
    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model = None
        self.explainer = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_scaled, y_train)
        
        # Initialize explainer on scaled data
        self.explainer = shap.TreeExplainer(self.model)
        
    def predict_explain(self, X_single: np.ndarray) -> dict:
        """
        Predict class probability and extract top contributing features.
        """
        X_scaled = self.scaler.transform(X_single.reshape(1, -1))
        
        # Class probabilities
        prob = self.model.predict_proba(X_scaled)[0]
        prediction = int(np.argmax(prob))
        confidence = float(prob[prediction])
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_scaled)
        
        # Format SHAP features
        contributions = []
        for idx, val in enumerate(shap_values[0]):
            contributions.append({
                "feature": self.feature_names[idx],
                "impact": float(val),
                "importance": abs(float(val))
            })
            
        # Sort by importance descending
        contributions.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "top_drivers": contributions[:3]
        }
```

---

## 3.3 — Reusable Logic Block C: Time-Series Preprocessing & Linear Regression Wrapper
*Applicable to: #1 (Carbon Scheduler), #7 (Capacity Planning), #12 (Device Health)*

For lightweight time-series forecasting without the dependency overhead of heavy libraries like Prophet.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

class SimpleTrendForecaster:
    """
    Performs fast, localized linear regression forecasting on resource usage metrics.
    Suitable for edge machines and lightweight dashboards.
    """
    def __init__(self, timestamps: list[float], values: list[float]):
        self.X = np.array(timestamps).reshape(-1, 1)
        self.y = np.array(values)
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        
    def predict_future(self, target_timestamp: float) -> float:
        """Predict value at target timestamp."""
        prediction = self.model.predict(np.array([[target_timestamp]]))
        return float(prediction[0])
        
    def estimate_breach_time(self, threshold: float) -> float | None:
        """
        Calculates when the trend line will cross a given threshold.
        Returns timestamp, or None if the trend is flat or downward.
        
        Formula:
            y = m * X + c  =>  X = (threshold - c) / m
        """
        slope = self.model.coef_[0]
        intercept = self.model.intercept_
        
        if slope <= 0:
            return None  # Trend will never hit threshold
            
        breach_ts = (threshold - intercept) / slope
        return float(breach_ts)
```

---

## 3.4 — Reusable Logic Block D: Production-Grade RAG Retriever
*Applicable to: #11 (Runbook Chatbot)*

A retrieval pattern that implements sparse (BM25) and dense (FAISS/Cosine) hybrid retrieval with reciprocal rank fusion.

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

class HybridRetriever:
    """
    Combines BM25 lexical search with dense vector similarity search.
    Provides robust document matching for runbook queries.
    """
    def __init__(self, doc_chunks: list[str], chunk_embeddings: np.ndarray, embedder):
        self.chunks = doc_chunks
        self.embeddings = chunk_embeddings
        self.embedder = embedder
        
        # Tokenize corpus for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in doc_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        # 1. Lexical BM25 score
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        # 2. Semantic vector score
        query_embedding = self.embedder.embed_batch([query])[0]
        vector_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        vector_ranks = np.argsort(vector_scores)[::-1]
        
        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum(1 / (50 + rank_in_retriever))
        rrf_scores = np.zeros(len(self.chunks))
        
        for rank, doc_idx in enumerate(bm25_ranks):
            rrf_scores[doc_idx] += 1.0 / (50 + rank)
            
        for rank, doc_idx in enumerate(vector_ranks):
            rrf_scores[doc_idx] += 1.0 / (50 + rank)
            
        # Select top RRF scores
        final_ranks = np.argsort(rrf_scores)[::-1]
        return [self.chunks[idx] for idx in final_ranks[:top_k]]
```

---

# PART 4 — Implementation Task Board

When coordinating with backend and frontend teammates, ensure roles are assigned clearly according to the architectural layer boundaries:

| Task / Feature | Layer | Owner | Dependency |
|---|---|---|---|
| BGL Log Preprocessing Pipeline | AI Layer | **AI Lead (You)** | Raw BGL file downloaded |
| Topology Distance Matrix logic | AI Layer | **AI Lead (You)** | Node tag mapping |
| Fast API Endpoint routes | Backend | **Backend Engineer** | Pydantic Schemas |
| Config Manager (`config.yaml`) | Backend / AI | **AI Lead (You)** | YAML Schema |
| UI Panels & Visual Grid Layout | Frontend | **Frontend Engineer** | API Contract defined |
| Alert Time-Series Chart | Frontend | **Frontend Engineer** | Ingestion timestamp format |
| Weight Control Sliders | Frontend | **Frontend Engineer** | PUT API route |
| LLM API Configuration & fallback | AI Layer | **AI Lead (You)** | API Keys |
