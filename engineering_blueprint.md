# Synergy 2026 — Engineering Blueprint

> **Purpose:** Exact AI logic, backend/frontend contracts, and team coordination guide
> **Owner:** AI Intelligence Layer Lead
> **Status:** Post-audit, revised architecture

---

# PART 1 — What We Are Building (Problem #10: Alert Correlation Engine)

## The One-Line Pitch

> A system that takes 1000+ raw monitoring alerts, clusters them into ~10 actual incidents, identifies the root cause of each, and summarizes them in plain English.

## Why This Wins

- **Demo moment:** "1247 alerts → 12 incidents" with a side-by-side visual
- **HPE relevance:** Their customers' #1 pain is alert fatigue
- **AI depth:** Embeddings + clustering + LLM — three distinct AI techniques, each justified
- **No vibe coding:** Every component has testable input/output, real math, real evaluation

---

# PART 2 — The AI Stack (Exact Technologies)

## What We Use and Why

| Component | Technology | Version | Why This, Not Something Else |
|-----------|-----------|---------|------------------------------|
| **Embeddings** | `sentence-transformers` / `all-MiniLM-L6-v2` | Latest | Runs locally. No API cost. 384-dim vectors. 100-200 sentences/sec on CPU. |
| **Similarity** | `sklearn.metrics.pairwise.cosine_similarity` | scikit-learn ≥1.3 | At <10K items, brute-force numpy beats any index (FAISS, Annoy). Zero setup. |
| **Clustering** | `sklearn.cluster.HDBSCAN` | scikit-learn ≥1.3 | No need to specify K. Handles noise points. Gives confidence scores. Now built into sklearn — one fewer dependency. |
| **LLM** | Google Gemini Flash via `google-genai` | Latest | Most generous free tier (~1500 req/day). Structured output support. |
| **LLM Backup** | Groq (Llama 3.3 70B) via `groq` | Latest | Fastest inference. 30 RPM free. Swap-in if Gemini is down. |
| **Backend** | FastAPI + Uvicorn | Latest | Async, auto-docs (Swagger), Pydantic validation built-in. |
| **Config** | PyYAML | Latest | Human-readable config for weights, params. |
| **Data** | In-memory Python dicts + JSON files | — | No database needed at hackathon scale. |

## What We Do NOT Use (and Why)

| Removed | Why Removed | What Replaces It |
|---------|-------------|-----------------|
| FAISS | Slower than numpy for <10K items (index build overhead) | `cosine_similarity` from sklearn |
| Redis | Adds Docker dependency for zero benefit at this scale | Python `dict` / `set` |
| MinHash LSH (`datasketch`) | Embedding cosine already handles fuzzy dedup. Three dedup methods is over-engineering. | Exact hash + cosine threshold |
| WebSocket | Debugging nightmare on demo day. Batch mode with a button click is safer. | REST API + "Correlate" button |
| Prometheus / MLflow | Nobody scrapes metrics at a hackathon | In-memory counters |
| LangChain / LlamaIndex | Not a RAG problem. No document retrieval needed. Direct API calls are simpler. | Raw `google-genai` calls |

---

# PART 3 — The Core AI Logic (Exact Algorithms)

> [!IMPORTANT]
> This section contains the **exact math and logic** for every AI component. No ambiguity. No hand-waving. Copy-paste into real code.

## 3.1 — Data Model (What an Alert Looks Like)

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class Alert(BaseModel):
    id: str                          # unique identifier (uuid4)
    timestamp: datetime              # when the alert fired
    source: str                      # monitoring system name
    severity: str                    # "critical" | "warning" | "info"
    service: str                     # e.g., "api-gateway", "db-primary"
    host: str                        # e.g., "prod-web-01"
    message: str                     # the alert text
    tags: list[str] = []             # e.g., ["production", "us-east-1"]

class Incident(BaseModel):
    id: str                          # cluster identifier
    alerts: list[Alert]              # member alerts
    root_cause: Alert                # the identified root cause alert
    summary: str                     # LLM-generated one-line summary
    confidence: float                # HDBSCAN cluster confidence (0-1)
    noise_reduction: float           # alerts_in_cluster / total_alerts
    engineer_hours_saved: float      # business metric
```

## 3.2 — Step 1: Deduplication

**Purpose:** Remove exact duplicate alerts before expensive embedding computation.

```python
def deduplicate(alerts: list[Alert]) -> list[Alert]:
    """Remove exact duplicate alerts based on normalized text hash."""
    seen_hashes = set()
    unique_alerts = []
    
    for alert in alerts:
        # Normalize: lowercase, strip whitespace, remove timestamps
        normalized = alert.message.strip().lower()
        # Remove dynamic parts (IPs, timestamps, UUIDs) with regex
        normalized = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IP>', normalized)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TS>', normalized)
        normalized = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<UUID>', normalized)
        
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_alerts.append(alert)
    
    return unique_alerts
```

**Why this works:** Exact hash is O(n). Zero false positives. The normalization catches alerts that differ only in timestamps/IPs but describe the same event.

## 3.3 — Step 2: Embedding

**Purpose:** Convert alert text into 384-dimensional vectors for semantic comparison.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self):
        # Load once, reuse. Downloads ~80MB on first run.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Input:  list of N alert message strings
        Output: numpy array of shape (N, 384)
        
        Performance: ~100-200 texts/sec on CPU
        For 1000 alerts: ~5-10 seconds
        """
        embeddings = self.model.encode(
            texts,
            batch_size=64,        # sweet spot for CPU
            show_progress_bar=False,
            normalize_embeddings=True  # L2-normalize for cosine = dot product
        )
        return embeddings  # shape: (N, 384)
```

**Critical detail:** `normalize_embeddings=True` means cosine similarity = simple dot product. This makes the similarity computation much faster.

## 3.4 — Step 3: Three-Signal Scoring

This is the **core innovation**. We score alert pairs on three independent signals and fuse them.

### Signal 1: Temporal Score

**Intuition:** Alerts that fire close together in time are more likely related.

```python
import math

def temporal_score(t1: datetime, t2: datetime, sigma: float = 300.0) -> float:
    """
    Gaussian decay based on time difference.
    
    Args:
        t1, t2: alert timestamps
        sigma: decay parameter in seconds (300 = 5 minutes half-life)
    
    Returns:
        float in [0, 1] where 1 = same time, 0 = far apart
    
    Formula:
        score = exp(-Δt² / (2 * σ²))
    
    Examples:
        Δt = 0 seconds  → score = 1.000
        Δt = 60 seconds → score = 0.980
        Δt = 300 seconds → score = 0.607  (one sigma)
        Δt = 600 seconds → score = 0.135
        Δt = 1800 seconds → score = 0.000  (effectively zero)
    """
    delta_seconds = abs((t1 - t2).total_seconds())
    return math.exp(-(delta_seconds ** 2) / (2 * sigma ** 2))
```

### Signal 2: Semantic Score

**Intuition:** Alerts with similar text describe similar problems.

```python
def semantic_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized embedding vectors.
    
    Since embeddings are pre-normalized, this is just a dot product.
    
    Args:
        emb1, emb2: 384-dim vectors (already L2-normalized)
    
    Returns:
        float in [-1, 1], practically [0, 1] for alert texts
        1 = identical meaning, 0 = unrelated
    
    Examples:
        "connection timeout on db-primary" vs 
        "database connection pool exhausted" → ~0.72
        
        "connection timeout on db-primary" vs 
        "DNS resolution failed for cdn.example.com" → ~0.18
    """
    return float(np.dot(emb1, emb2))
```

### Signal 3: Topological Score

**Intuition:** Alerts from the same service/host/tags are more likely related.

```python
def topological_score(alert1: Alert, alert2: Alert) -> float:
    """
    Jaccard similarity on the combined set of service, host, and tags.
    
    Args:
        alert1, alert2: Alert objects with service, host, tags fields
    
    Returns:
        float in [0, 1] where 1 = identical topology, 0 = no overlap
    
    Formula:
        set1 = {service1, host1} ∪ tags1
        set2 = {service2, host2} ∪ tags2
        jaccard = |set1 ∩ set2| / |set1 ∪ set2|
    
    Examples:
        Same service, same host → ~0.6-1.0
        Same service, diff host → ~0.3-0.5
        Diff service, diff host → ~0.0-0.1
    """
    set1 = {alert1.service, alert1.host} | set(alert1.tags)
    set2 = {alert2.service, alert2.host} | set(alert2.tags)
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    return intersection / union
```

## 3.5 — Step 4: Fusion Matrix

**Purpose:** Combine all three scores into a single n×n distance matrix for clustering.

```python
from sklearn.metrics.pairwise import cosine_similarity

def compute_fusion_matrix(
    alerts: list[Alert],
    embeddings: np.ndarray,
    weights: dict  # {"temporal": 0.3, "semantic": 0.5, "topological": 0.2}
) -> np.ndarray:
    """
    Compute n×n fused distance matrix.
    
    Returns:
        numpy array of shape (n, n) where entry [i,j] is the 
        DISTANCE between alerts i and j. Range [0, 1].
        0 = identical, 1 = completely unrelated.
    """
    n = len(alerts)
    w_t = weights["temporal"]
    w_s = weights["semantic"]
    w_p = weights["topological"]
    
    # --- Semantic scores (batch computation — FAST) ---
    # cosine_similarity returns (n, n) matrix, all pairs at once
    semantic_matrix = cosine_similarity(embeddings)  # shape (n, n)
    # Clip to [0, 1] (cosine can be slightly negative)
    semantic_matrix = np.clip(semantic_matrix, 0.0, 1.0)
    
    # --- Temporal & Topological scores (pairwise loop) ---
    temporal_matrix = np.zeros((n, n))
    topo_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            t_score = temporal_score(alerts[i].timestamp, alerts[j].timestamp)
            p_score = topological_score(alerts[i], alerts[j])
            
            temporal_matrix[i, j] = t_score
            temporal_matrix[j, i] = t_score
            topo_matrix[i, j] = p_score
            topo_matrix[j, i] = p_score
    
    # Fill diagonal with 1.0 (self-similarity)
    np.fill_diagonal(temporal_matrix, 1.0)
    np.fill_diagonal(topo_matrix, 1.0)
    
    # --- Fuse into similarity matrix ---
    similarity_matrix = (
        w_t * temporal_matrix + 
        w_s * semantic_matrix + 
        w_p * topo_matrix
    )
    
    # Convert similarity → distance
    distance_matrix = 1.0 - similarity_matrix
    
    # Ensure valid distance matrix (symmetric, zero diagonal, non-negative)
    np.fill_diagonal(distance_matrix, 0.0)
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    
    return distance_matrix
```

> [!WARNING]
> **Scaling note:** The pairwise loop is O(n²). For n=1000, that's 500K pairs. This runs in ~2-5 seconds in Python. For n=5000, use numpy vectorization or precompute timestamps as floats. This is fine for the hackathon.

## 3.6 — Step 5: Clustering (HDBSCAN)

```python
from sklearn.cluster import HDBSCAN

def cluster_alerts(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster alerts using HDBSCAN on the precomputed distance matrix.
    
    Args:
        distance_matrix: n×n distance matrix from fusion step
        min_cluster_size: minimum alerts to form an incident (3 is good default)
        min_samples: core point density parameter
    
    Returns:
        labels: array of cluster labels. -1 = noise (unclustered alert)
        probabilities: array of membership confidence scores [0, 1]
    
    Example output for 500 alerts, 4 real incidents:
        labels: [0, 0, 0, 1, 1, -1, 2, 2, 2, 2, 3, ...]
        probabilities: [0.95, 0.88, 0.72, 0.91, ...]
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',          # we provide our own distance matrix
        cluster_selection_method='eom'  # excess of mass (default, good)
    )
    clusterer.fit(distance_matrix)
    
    return clusterer.labels_, clusterer.probabilities_
```

**What HDBSCAN gives us for free:**
- Cluster labels (which incident each alert belongs to)
- `-1` labels for noise (alerts that don't fit any cluster)
- Probability scores (confidence that an alert truly belongs to its cluster)
- No need to guess the number of clusters

## 3.7 — Step 6: Root Cause Identification

```python
SEVERITY_MAP = {"critical": 1.0, "warning": 0.6, "info": 0.2}

def identify_root_cause(
    cluster_alerts: list[Alert],
    w_time: float = 0.5,
    w_severity: float = 0.3,
    w_centrality: float = 0.2
) -> Alert:
    """
    Identify the most likely root cause alert in a cluster.
    
    Heuristic: The root cause is the alert that:
      1. Fired EARLIEST in the cluster (cascade starts here)
      2. Has the HIGHEST severity (critical > warning > info)  
      3. Is most CENTRAL (most similar to other alerts in the cluster)
    
    Formula:
        root_score(i) = w1 * time_rank(i) + w2 * severity(i) + w3 * centrality(i)
    
    All components are normalized to [0, 1].
    """
    n = len(cluster_alerts)
    if n == 1:
        return cluster_alerts[0]
    
    # 1. Time rank: earliest alert gets score 1.0, latest gets 0.0
    sorted_by_time = sorted(range(n), key=lambda i: cluster_alerts[i].timestamp)
    time_scores = [0.0] * n
    for rank, idx in enumerate(sorted_by_time):
        time_scores[idx] = 1.0 - (rank / (n - 1)) if n > 1 else 1.0
    
    # 2. Severity score: direct mapping
    severity_scores = [
        SEVERITY_MAP.get(a.severity.lower(), 0.3) for a in cluster_alerts
    ]
    
    # 3. Centrality: how many other alerts share the same service
    service_counts = {}
    for a in cluster_alerts:
        service_counts[a.service] = service_counts.get(a.service, 0) + 1
    max_count = max(service_counts.values())
    centrality_scores = [
        service_counts[a.service] / max_count for a in cluster_alerts
    ]
    
    # Composite score
    composite = [
        w_time * time_scores[i] + 
        w_severity * severity_scores[i] + 
        w_centrality * centrality_scores[i]
        for i in range(n)
    ]
    
    root_idx = max(range(n), key=lambda i: composite[i])
    return cluster_alerts[root_idx]
```

## 3.8 — Step 7: LLM Summarization (with Fallback)

```python
import google.generativeai as genai

class Summarizer:
    def __init__(self, api_key: str, backup_api_key: str = None):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.backup_key = backup_api_key
    
    def summarize_cluster(
        self, 
        alerts: list[Alert], 
        root_cause: Alert
    ) -> str:
        """
        Generate a one-line incident summary from a cluster of alerts.
        
        Fallback chain:
          1. Try Gemini Flash
          2. Try Groq (if backup key exists)
          3. Use template-based summary (always works, no API needed)
        """
        prompt = self._build_prompt(alerts, root_cause)
        
        # Attempt 1: Gemini
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            if len(summary) > 10:  # basic sanity check
                return summary
        except Exception:
            pass
        
        # Attempt 2: Groq backup
        if self.backup_key:
            try:
                return self._call_groq(prompt)
            except Exception:
                pass
        
        # Attempt 3: Template fallback (ALWAYS works)
        return self._template_summary(alerts, root_cause)
    
    def _build_prompt(self, alerts: list[Alert], root_cause: Alert) -> str:
        # Take at most 15 representative alerts to keep prompt small
        sample = alerts[:15] if len(alerts) > 15 else alerts
        alert_texts = "\n".join([
            f"- [{a.severity}] {a.service}/{a.host}: {a.message}" 
            for a in sample
        ])
        
        return f"""You are an infrastructure incident analyst.

Given these {len(alerts)} related monitoring alerts, write exactly ONE sentence 
that describes the incident — what happened, what was affected, and the likely root cause.

Root cause alert: [{root_cause.severity}] {root_cause.service}/{root_cause.host}: {root_cause.message}

Sample alerts:
{alert_texts}

Respond with ONLY the one-sentence summary. No preamble."""

    def _template_summary(self, alerts: list[Alert], root_cause: Alert) -> str:
        """Guaranteed-working template. No API needed."""
        services = list(set(a.service for a in alerts))
        services_str = ", ".join(services[:4])
        if len(services) > 4:
            services_str += f" and {len(services) - 4} more"
        
        return (
            f"{root_cause.message} on {root_cause.service}/{root_cause.host} "
            f"caused {len(alerts)} related alerts across {services_str}."
        )
```

## 3.9 — The Complete Pipeline (Orchestrator)

```python
class CorrelationEngine:
    """
    The single entry point for the AI layer.
    
    Usage:
        engine = CorrelationEngine(config)
        incidents = engine.correlate(raw_alerts)
    
    This is what the backend calls. It returns clean data.
    The backend never touches embeddings, clustering, or LLM directly.
    """
    
    def __init__(self, config: dict):
        self.embedder = Embedder()
        self.summarizer = Summarizer(
            api_key=config["gemini_api_key"],
            backup_api_key=config.get("groq_api_key")
        )
        self.weights = config["fusion_weights"]
        self.hdbscan_params = config["hdbscan"]
    
    def correlate(self, alerts: list[Alert]) -> list[Incident]:
        """
        MAIN METHOD. Takes raw alerts, returns structured incidents.
        
        Pipeline: dedup → embed → score → fuse → cluster → root_cause → summarize
        """
        import time
        start = time.time()
        
        # Step 1: Dedup
        unique_alerts = deduplicate(alerts)
        
        # Step 2: Embed
        texts = [a.message for a in unique_alerts]
        embeddings = self.embedder.embed_batch(texts)
        
        # Step 3-4: Score + Fuse
        distance_matrix = compute_fusion_matrix(
            unique_alerts, embeddings, self.weights
        )
        
        # Step 5: Cluster
        labels, probabilities = cluster_alerts(
            distance_matrix,
            min_cluster_size=self.hdbscan_params["min_cluster_size"],
            min_samples=self.hdbscan_params["min_samples"]
        )
        
        # Step 6-7: Build incidents
        incidents = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise label
        
        for label in sorted(unique_labels):
            mask = labels == label
            cluster_indices = np.where(mask)[0]
            cluster_alerts_list = [unique_alerts[i] for i in cluster_indices]
            cluster_confidence = float(np.mean(probabilities[mask]))
            
            # Root cause
            root = identify_root_cause(cluster_alerts_list)
            
            # LLM summary
            summary = self.summarizer.summarize_cluster(
                cluster_alerts_list, root
            )
            
            incident = Incident(
                id=f"INC-{label:03d}",
                alerts=cluster_alerts_list,
                root_cause=root,
                summary=summary,
                confidence=round(cluster_confidence, 3),
                noise_reduction=round(len(cluster_alerts_list) / len(alerts), 4),
                engineer_hours_saved=round(len(cluster_alerts_list) * 2 / 60, 1)
            )
            incidents.append(incident)
        
        elapsed = round(time.time() - start, 2)
        
        return incidents, {
            "total_alerts": len(alerts),
            "after_dedup": len(unique_alerts),
            "incidents_found": len(incidents),
            "noise_alerts": int(np.sum(labels == -1)),
            "processing_time_seconds": elapsed,
            "noise_reduction_pct": round(
                (1 - len(incidents) / max(len(unique_alerts), 1)) * 100, 1
            )
        }
```

---

# PART 4 — Backend / Frontend Architecture & Contracts

## The Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   FRONTEND (Built by Frontend Dev)                         │
│   • HTML/CSS/JS  OR  Streamlit                             │
│   • Calls backend REST API                                 │
│   • Renders results visually                               │
│                                                             │
│   Responsibility: UI, charts, user interaction             │
│   Does NOT: touch AI, embeddings, clustering, LLM          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   BACKEND (Built by Backend Dev + AI Lead)                 │
│   • FastAPI application                                    │
│   • Receives HTTP requests from frontend                   │
│   • Calls AI layer, returns structured JSON                │
│                                                             │
│   Responsibility: API routing, validation, CORS, errors    │
│   Does NOT: render HTML, do heavy AI computation           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   AI LAYER (Built by AI Lead — YOU)                        │
│   • CorrelationEngine class                                │
│   • Embeddings, scoring, clustering, LLM                   │
│   • Exposes ONE method: engine.correlate(alerts)           │
│                                                             │
│   Responsibility: All intelligence                         │
│   Does NOT: handle HTTP, render UI, manage routes          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The Connection Between Layers

**The golden rule:** The backend imports the AI layer as a Python class. The frontend talks to the backend via HTTP JSON.

```python
# main.py — HOW BACKEND CONNECTS TO AI LAYER
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.engine import CorrelationEngine   # ← AI layer
from models.alert import Alert
import yaml

app = FastAPI(title="Alert Correlation Engine")

# CORS so frontend can call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # for hackathon. restrict in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config and init AI engine ONCE at startup
with open("config.yaml") as f:
    config = yaml.safe_load(f)

engine = CorrelationEngine(config)

# ---- API ENDPOINTS (what frontend calls) ----

@app.post("/api/alerts/correlate")
async def correlate_alerts(alerts: list[Alert]):
    """The main endpoint. Frontend sends alerts, gets incidents back."""
    incidents, metrics = engine.correlate(alerts)
    return {
        "incidents": [inc.model_dump() for inc in incidents],
        "metrics": metrics
    }

@app.put("/api/config/weights")
async def update_weights(weights: dict):
    """Frontend sliders send new weights here."""
    engine.weights = weights
    return {"status": "updated", "weights": weights}

@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

## Exact API Contracts (Hand This to Your Team)

### Contract 1: Correlate Alerts

**Tell your backend/frontend devs:** "Send a POST to `/api/alerts/correlate` with this JSON. Get back incidents."

```
POST /api/alerts/correlate
Content-Type: application/json

REQUEST BODY:
[
  {
    "id": "alert-001",
    "timestamp": "2026-07-15T14:30:00Z",
    "source": "prometheus",
    "severity": "critical",
    "service": "db-primary",
    "host": "prod-db-01",
    "message": "Connection pool exhausted: 0/100 connections available",
    "tags": ["production", "us-east-1", "database"]
  },
  {
    "id": "alert-002",
    "timestamp": "2026-07-15T14:31:15Z",
    "source": "prometheus",
    "severity": "warning",
    "service": "api-gateway",
    "host": "prod-api-03",
    "message": "Request timeout after 30s waiting for database response",
    "tags": ["production", "us-east-1", "api"]
  }
  // ... more alerts
]

RESPONSE (200 OK):
{
  "incidents": [
    {
      "id": "INC-000",
      "alerts": [ ... ],           // full alert objects in this cluster
      "root_cause": {              // the identified root cause alert
        "id": "alert-001",
        "severity": "critical",
        "service": "db-primary",
        "message": "Connection pool exhausted: 0/100 connections available"
      },
      "summary": "Database connection pool exhaustion on db-primary/prod-db-01 caused cascading API timeouts across gateway, auth, and dashboard services.",
      "confidence": 0.912,
      "noise_reduction": 0.358,
      "engineer_hours_saved": 14.9
    }
  ],
  "metrics": {
    "total_alerts": 1247,
    "after_dedup": 1103,
    "incidents_found": 12,
    "noise_alerts": 47,
    "processing_time_seconds": 4.82,
    "noise_reduction_pct": 98.9
  }
}
```

### Contract 2: Update Fusion Weights

**Tell your frontend dev:** "When the user moves the slider, send this."

```
PUT /api/config/weights
Content-Type: application/json

REQUEST BODY:
{
  "temporal": 0.3,
  "semantic": 0.5,
  "topological": 0.2
}

RESPONSE (200 OK):
{
  "status": "updated",
  "weights": {"temporal": 0.3, "semantic": 0.5, "topological": 0.2}
}
```

> After updating weights, the frontend should call `/api/alerts/correlate` again with the same alerts to see the new clustering.

### Contract 3: Health Check

```
GET /api/health

RESPONSE: {"status": "ok"}
```

## What to Tell Each Teammate

### Tell the Backend Dev

> "Install FastAPI and Uvicorn. I'll give you the `CorrelationEngine` class — you just import it and call `engine.correlate(alerts)`. It returns a list of `Incident` objects and a metrics dict. You wire that into the API endpoints. I'll write the Pydantic models. You handle CORS, error handling, and the Swagger docs. The config comes from `config.yaml`. Don't worry about embeddings, clustering, or LLM — I handle all of that inside the engine."

### Tell the Frontend Dev

> "The backend gives you two endpoints: POST `/api/alerts/correlate` (send alerts, get incidents) and PUT `/api/config/weights` (send slider values). I'll give you sample JSON files for each demo scenario — you can load those and POST them. Build two panels: left side shows raw alerts (scrollable list, color by severity), right side shows incident cards (each card = cluster summary + root cause + member count + confidence). Add a metrics bar (total alerts, incidents found, noise reduction %, processing time). Add three sliders (temporal, semantic, topological — each 0 to 1) and a 'Re-correlate' button. Add a timeline chart (alerts per minute on x-axis, count on y-axis)."

---

# PART 5 — config.yaml (The Single Source of Truth)

```yaml
# ---- AI Configuration ----
gemini_api_key: "YOUR_GEMINI_KEY"     # from aistudio.google.com
groq_api_key: "YOUR_GROQ_KEY"        # from console.groq.com (backup)

# Embedding model
embedding_model: "all-MiniLM-L6-v2"

# Fusion weights (must sum to 1.0)
fusion_weights:
  temporal: 0.3
  semantic: 0.5
  topological: 0.2

# HDBSCAN parameters
hdbscan:
  min_cluster_size: 3        # minimum alerts to form an incident
  min_samples: 2             # core sample density

# Temporal scoring
temporal_sigma: 300          # Gaussian decay sigma in seconds (5 min)

# Root cause weights
root_cause_weights:
  time: 0.5
  severity: 0.3
  centrality: 0.2

# Business metrics
avg_triage_minutes_per_alert: 2   # for engineer-hours-saved calculation
```

---

# PART 6 — Generalized AI Patterns (Applicable to ANY of the 15 Problems)

> These patterns are reusable across problems. If you switch from #10 to any other problem, these still apply.

## Pattern 1: LLM Integration with Fallback Chain

**Applies to:** Problems #2, #3, #4, #6, #7, #8, #10, #11, #14, #15

```python
class LLMClient:
    """
    Reusable LLM client with automatic fallback.
    Works for ANY problem that needs LLM text generation.
    """
    def __init__(self, gemini_key: str, groq_key: str = None):
        self.gemini_key = gemini_key
        self.groq_key = groq_key
    
    def generate(self, prompt: str, fallback_text: str = "Generation failed") -> str:
        """
        Try Gemini → Try Groq → Return fallback.
        NEVER crashes. ALWAYS returns a string.
        """
        # Attempt 1: Gemini Flash
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text:
                return text
        except Exception as e:
            print(f"[LLM] Gemini failed: {e}")
        
        # Attempt 2: Groq
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
            except Exception as e:
                print(f"[LLM] Groq failed: {e}")
        
        # Attempt 3: Template fallback
        return fallback_text
```

**Usage across problems:**
- **#6 Postmortem:** `llm.generate(postmortem_prompt, fallback_text=template_postmortem)`
- **#8 Handoff:** `llm.generate(handoff_prompt, fallback_text=template_handoff)`
- **#15 Infra Docs:** `llm.generate(doc_prompt, fallback_text=template_doc)`

## Pattern 2: Embedding + Similarity (Reusable Everywhere)

**Applies to:** Problems #4, #10, #11, #14

```python
class SimilarityEngine:
    """
    Reusable embedding-based similarity computation.
    Works for: alert correlation, log classification, runbook search, config comparison.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
    
    def compute_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """
        Input: N texts
        Output: N×N similarity matrix (float, 0 to 1)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return cosine_similarity(embeddings)
    
    def find_most_similar(self, query: str, corpus: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """
        Find top-k most similar items to a query.
        Returns: list of (index, score) tuples, sorted by score desc.
        
        Use cases:
          - Runbook chatbot (#11): find relevant runbook chunks
          - Log classifier (#4): find similar labeled examples
          - Config drift (#14): find similar past changes
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        corpus_embs = self.model.encode(corpus, normalize_embeddings=True, batch_size=64)
        
        scores = cosine_similarity(query_emb, corpus_embs)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
```

## Pattern 3: Structured LLM Output (Pydantic)

**Applies to:** Problems #2, #6, #7, #8, #10, #14, #15

```python
from pydantic import BaseModel

class PostmortemOutput(BaseModel):
    """Example: enforce LLM output structure for Problem #6."""
    title: str
    timeline: list[str]
    root_cause: str
    impact: str
    remediation_steps: list[str]
    lessons_learned: list[str]

def generate_structured(llm: LLMClient, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """
    Ask LLM for structured JSON output, validate with Pydantic.
    
    If LLM returns invalid JSON, retry once with a corrective prompt.
    If still invalid, raise an error (caller should have a template fallback).
    """
    import json
    
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    full_prompt = f"""{prompt}

Respond ONLY with valid JSON matching this schema:
{schema_json}

No markdown. No explanation. Just the JSON object."""

    raw = llm.generate(full_prompt, fallback_text="{}")
    
    # Clean common LLM output issues
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    
    try:
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception:
        # Retry with correction
        retry_prompt = f"Your previous response was not valid JSON. Please try again.\n\n{full_prompt}"
        raw2 = llm.generate(retry_prompt, fallback_text="{}")
        raw2 = raw2.strip().strip("`").strip()
        if raw2.startswith("json"):
            raw2 = raw2[4:].strip()
        data2 = json.loads(raw2)
        return schema.model_validate(data2)
```

## Pattern 4: ML Classification with Explainability (SHAP)

**Applies to:** Problems #5, #9, #13

```python
import xgboost as xgb
import shap

class ExplainableClassifier:
    """
    Train XGBoost + get SHAP explanations.
    Reusable for: deployment risk (#9), SLA breach (#5), traffic anomaly (#13).
    """
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        self.feature_names = feature_names
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=42
        )
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
    
    def predict_with_explanation(self, X_single: np.ndarray) -> dict:
        """
        Predict + explain WHY for a single sample.
        
        Returns:
          {
            "prediction": "high",
            "probability": 0.87,
            "explanations": [
              {"feature": "time_of_day", "impact": 0.3, "direction": "increases risk"},
              {"feature": "lines_changed", "impact": 0.25, "direction": "increases risk"},
              ...
            ]
          }
        """
        proba = self.model.predict_proba(X_single.reshape(1, -1))[0]
        pred_class = int(np.argmax(proba))
        
        shap_values = self.explainer.shap_values(X_single.reshape(1, -1))
        
        explanations = []
        for i, name in enumerate(self.feature_names):
            sv = float(shap_values[0][i]) if isinstance(shap_values, np.ndarray) else float(shap_values[pred_class][0][i])
            explanations.append({
                "feature": name,
                "impact": round(abs(sv), 4),
                "direction": "increases risk" if sv > 0 else "decreases risk"
            })
        
        explanations.sort(key=lambda x: x["impact"], reverse=True)
        
        class_map = {0: "low", 1: "medium", 2: "high"}
        return {
            "prediction": class_map.get(pred_class, str(pred_class)),
            "probability": round(float(max(proba)), 3),
            "explanations": explanations[:5]  # top 5 factors
        }
```

## Pattern 5: Synthetic Data Generation

**Applies to:** ALL problems that need realistic test data

```python
import random
from datetime import datetime, timedelta

def generate_incident_scenario(
    base_time: datetime,
    root_cause_msg: str,
    root_cause_service: str,
    cascade_services: list[str],
    total_alerts: int = 300,
    duration_minutes: int = 15
) -> list[dict]:
    """
    Generate a realistic cascading incident.
    
    The root cause fires first. Downstream services fail 1-5 minutes later.
    Alerts accumulate with realistic timing and severity distribution.
    
    This pattern works for ANY monitoring/ops problem statement.
    """
    alerts = []
    severities = ["critical", "warning", "info"]
    severity_weights = [0.15, 0.35, 0.50]
    
    # Root cause alert (fires first)
    alerts.append({
        "id": f"alert-{len(alerts):04d}",
        "timestamp": base_time.isoformat(),
        "source": "prometheus",
        "severity": "critical",
        "service": root_cause_service,
        "host": f"prod-{root_cause_service}-01",
        "message": root_cause_msg,
        "tags": ["production", "us-east-1"]
    })
    
    # Cascade: downstream services fail with delay
    for i in range(total_alerts - 1):
        delay_seconds = random.randint(30, duration_minutes * 60)
        service = random.choice(cascade_services)
        severity = random.choices(severities, severity_weights)[0]
        
        cascade_messages = [
            f"Request timeout waiting for {root_cause_service} response",
            f"Health check failed: dependency {root_cause_service} unreachable",
            f"Error rate exceeded threshold: 5xx responses at 45%",
            f"Connection refused when reaching {root_cause_service}",
            f"Circuit breaker OPEN for {root_cause_service} calls",
        ]
        
        alerts.append({
            "id": f"alert-{len(alerts):04d}",
            "timestamp": (base_time + timedelta(seconds=delay_seconds)).isoformat(),
            "source": random.choice(["prometheus", "grafana", "pagerduty"]),
            "severity": severity,
            "service": service,
            "host": f"prod-{service}-{random.randint(1, 5):02d}",
            "message": random.choice(cascade_messages),
            "tags": ["production", "us-east-1", service.split("-")[0]]
        })
    
    return alerts
```

## Pattern 6: FastAPI Backend Skeleton (Works for Any Problem)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="[Problem Name] Engine",
    description="Synergy 2026 HPE Hackathon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Initialize AI layer at startup ----
@app.on_event("startup")
async def startup():
    # Load model, config, etc. ONCE
    app.state.engine = YourAIEngine(config)

# ---- Endpoints follow the same pattern ----
@app.post("/api/process")
async def process(input_data: YourInputModel):
    try:
        result = app.state.engine.process(input_data)
        return {"result": result, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": app.state.engine is not None}
```

---

# PART 7 — Anti-Vibe-Coding Practices

> [!CAUTION]
> **Vibe coding** = writing code that "looks right" without testing if it actually works. Most hackathon teams do this. We do NOT.

## Rule 1: Every AI Function Has Expected Input/Output

```python
# BAD (vibe coding):
def process_alerts(data):
    results = model.predict(data)
    return results

# GOOD (our approach):
def process_alerts(alerts: list[Alert]) -> tuple[list[Incident], dict]:
    """
    Input: list of Alert objects (validated by Pydantic)
    Output: tuple of (incidents list, metrics dict)
    
    Guarantees:
    - Always returns a tuple, never None
    - incidents list may be empty but never None
    - metrics dict always has keys: total_alerts, incidents_found, processing_time_seconds
    """
```

## Rule 2: Every AI Function Has a Unit Test

```python
def test_temporal_score():
    from datetime import datetime
    t1 = datetime(2026, 7, 15, 14, 0, 0)
    t2 = datetime(2026, 7, 15, 14, 0, 0)     # same time
    assert temporal_score(t1, t2) == 1.0       # must be 1.0
    
    t3 = datetime(2026, 7, 15, 14, 5, 0)      # 5 min later
    score = temporal_score(t1, t3, sigma=300)
    assert 0.5 < score < 0.7                   # should be ~0.607
    
    t4 = datetime(2026, 7, 15, 15, 0, 0)      # 1 hour later
    score = temporal_score(t1, t4, sigma=300)
    assert score < 0.01                        # effectively zero

def test_dedup_removes_exact_duplicates():
    alerts = [
        Alert(id="1", timestamp="2026-07-15T14:00:00Z", source="p", severity="warning", 
              service="api", host="h1", message="Connection timeout"),
        Alert(id="2", timestamp="2026-07-15T14:01:00Z", source="p", severity="warning", 
              service="api", host="h1", message="Connection timeout"),  # same message
    ]
    result = deduplicate(alerts)
    assert len(result) == 1  # duplicate removed

def test_pipeline_end_to_end():
    """The most important test. If this passes, your demo works."""
    from data.generator import generate_incident_scenario
    from datetime import datetime
    
    alerts = generate_incident_scenario(
        base_time=datetime(2026, 7, 15, 14, 0, 0),
        root_cause_msg="Connection pool exhausted",
        root_cause_service="db-primary",
        cascade_services=["api-gateway", "auth-service", "dashboard"],
        total_alerts=100
    )
    
    engine = CorrelationEngine(test_config)
    incidents, metrics = engine.correlate([Alert(**a) for a in alerts])
    
    # Basic sanity checks
    assert len(incidents) >= 1           # at least one incident found
    assert metrics["total_alerts"] == 100
    assert metrics["processing_time_seconds"] < 30  # not hanging
    assert all(inc.summary for inc in incidents)     # summaries not empty
    assert all(inc.root_cause for inc in incidents)  # root causes identified
```

## Rule 3: Every External Call Has a Timeout and Fallback

```python
# BAD:
response = genai.generate_content(prompt)    # can hang forever

# GOOD:
import signal

def call_with_timeout(func, args, timeout=10, fallback=None):
    """Call a function with a timeout. Return fallback if it hangs or crashes."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, Exception):
            return fallback
```

## Rule 4: The Demo NEVER Crashes

```python
# In main.py — global exception handler
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch ALL exceptions. Return a clean error. Never show a stack trace to frontend."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal processing error",
            "detail": str(exc)[:200],  # truncate to prevent data leak
            "status": "error"
        }
    )
```

---

# PART 8 — Project File Structure (Final)

```
alert_engine/
├── main.py                        # FastAPI app entry point
├── config.yaml                    # ALL configuration (weights, keys, params)
├── requirements.txt               # Pinned dependency versions
├── README.md                      # Setup + architecture + metrics
│
├── models/                        # Pydantic data models
│   ├── __init__.py
│   ├── alert.py                   # Alert schema
│   └── incident.py                # Incident schema
│
├── core/                          # AI LAYER (your code)
│   ├── __init__.py
│   ├── engine.py                  # CorrelationEngine (orchestrator)
│   ├── embedder.py                # Sentence-transformer wrapper
│   ├── scorers.py                 # temporal, semantic, topological scorers
│   ├── fusion.py                  # Multi-signal fusion matrix
│   ├── clusterer.py               # HDBSCAN wrapper
│   ├── root_cause.py              # Root cause heuristic
│   ├── summarizer.py              # LLM summarization + fallback
│   └── dedup.py                   # Exact-hash deduplication
│
├── api/                           # BACKEND LAYER (backend dev's code)
│   ├── __init__.py
│   └── routes.py                  # FastAPI endpoints
│
├── data/                          # Test data
│   ├── generator.py               # Synthetic scenario generator
│   └── scenarios/
│       ├── db_outage.json         # Pre-generated scenario 1
│       ├── dns_failure.json       # Pre-generated scenario 2
│       └── ddos_attack.json       # Pre-generated scenario 3
│
├── tests/                         # Tests (anti-vibe-coding)
│   ├── test_scorers.py
│   ├── test_clustering.py
│   ├── test_dedup.py
│   ├── test_pipeline.py           # End-to-end integration test
│   └── test_api.py
│
└── dashboard/                     # FRONTEND LAYER (frontend dev's code)
    ├── index.html                 # OR streamlit app.py
    └── ...
```

---

# PART 9 — Quick Reference Card

## The Pipeline in 7 Steps

```
Raw Alerts (JSON list)
    │
    ▼
[1. DEDUP] ─── hash(normalize(text)) ─── remove exact duplicates
    │
    ▼
[2. EMBED] ─── all-MiniLM-L6-v2 ─── texts → 384-dim vectors
    │
    ▼
[3. SCORE] ─── temporal (Gaussian decay on Δt)
           ─── semantic (cosine on embeddings)
           ─── topological (Jaccard on service/host/tags)
    │
    ▼
[4. FUSE]  ─── distance = 1 - (0.3·temporal + 0.5·semantic + 0.2·topological)
    │
    ▼
[5. CLUSTER] ─── HDBSCAN(metric='precomputed', min_cluster_size=3)
    │
    ▼
[6. ROOT CAUSE] ─── score = 0.5·time_rank + 0.3·severity + 0.2·centrality
    │
    ▼
[7. SUMMARIZE] ─── Gemini Flash → one-line summary per cluster
                   (fallback: template if API fails)
```

## Key Numbers for the Demo

| Metric | Target | How to Achieve |
|--------|--------|---------------|
| Processing time (1000 alerts) | < 15 seconds | Pre-compute embeddings; batch cosine similarity |
| Noise reduction | > 85% | Tune HDBSCAN min_cluster_size |
| Root cause accuracy | > 75% | Tune root cause weights on synthetic scenarios |
| LLM summary quality | Reads like a human wrote it | Good prompt engineering + structured output |
| Dashboard load time | < 2 seconds | Pre-compute results; serve cached JSON |

## Emergency Procedures (Demo Day)

| If This Happens | Do This |
|-----------------|---------|
| Gemini API is down | System auto-falls back to template summaries. Mention: "We built graceful degradation." |
| Clustering looks wrong | Switch to a different pre-built scenario JSON. |
| Frontend crashes | Open Swagger docs at `/docs`. Run the demo through the API directly. |
| Everything crashes | Play the backup video recording. |
| Judge asks about scaling | "In-memory handles 10K alerts. For production: swap numpy for FAISS, add Redis for caching, containerize with K8s. The architecture is designed for it." |
