# IMPLEMENTATION SPEC — BGL Alert Correlation & Deduplication Engine

> **PURPOSE:** Give this entire document to any coding LLM. It will produce a complete, working, demo-ready application based on the **original Loghub BGL supercomputer log dataset**, optimized for maximum clustering accuracy.
> **CONSTRAINT:** All code must be Python 3.11+. No paid APIs. Must run locally.

---

# SYSTEM OVERVIEW

Build an alert correlation engine that:
1. Automatically downloads the original `BGL_2k.log` dataset from Loghub's GitHub repository.
2. Parses raw BGL logs into structured objects with zero data loss, using log levels and alert tags.
3. Computes a physical topology distance metric from BlueGene/L coordinate formats (`Rxx-Mxx-Nxx`).
4. Groups alerts using density-based clustering (HDBSCAN) on a fused distance matrix (temporal, semantic, topological).
5. Identifies the root cause alert based on timestamp, severity, and node centrality.
6. Evaluates clustering quality against ground-truth labels using **Normalized Mutual Information (NMI)**, **Adjusted Rand Index (ARI)**, and **Silhouette Score**.
7. Provides a **Weight Optimizer** that searches for the optimal weights ($w_t, w_s, w_p$) to maximize clustering accuracy (NMI).
8. Exposes these operations via FastAPI and visualizes them in a Streamlit dashboard.

---

# FILE STRUCTURE

```
alert_engine/
├── main.py                        # FastAPI entry point
├── config.yaml                    # System configuration
├── requirements.txt               # Pinned dependencies
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── alert.py                   # Pydantic Alert model (captures BGL schema)
│   └── incident.py                # Pydantic Incident & Metric models
│
├── core/
│   ├── __init__.py
│   ├── dedup.py                   # Exact-hash text deduplication
│   ├── embedder.py                # Sentence-transformer wrapper
│   ├── scorers.py                 # temporal, semantic, topological scorers
│   ├── fusion.py                  # multi-signal distance matrix builder
│   ├── clusterer.py               # HDBSCAN precomputed clustering
│   ├── root_cause.py              # Root cause heuristic
│   ├── summarizer.py              # Gemini/Groq + template summarizer
│   ├── evaluator.py               # NMI, ARI, Silhouette score calculator
│   ├── optimizer.py               # Grid search weight optimizer
│   └── engine.py                  # Orchestrator (CorrelationEngine)
│
├── data/
│   ├── __init__.py
│   ├── downloader.py              # Automatic dataset downloader
│   └── bgl_parser.py              # BGL raw log lines parser
│
├── api/
│   ├── __init__.py
│   └── routes.py                  # FastAPI router (correlation + optimization)
│
└── dashboard/
    └── app.py                     # Streamlit application
```

---

# DEPENDENCIES (`requirements.txt`)

```
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.9.0
sentence-transformers==3.0.0
scikit-learn==1.5.0
numpy==1.26.0
google-genai==1.0.0
groq==0.9.0
pyyaml==6.0.2
streamlit==1.38.0
httpx==0.27.0
pytest==8.3.0
```

---

# CONFIGURATION (`config.yaml`)

```yaml
# API keys
gemini_api_key: "YOUR_GEMINI_KEY"
groq_api_key: "YOUR_GROQ_KEY"

# AI Pipeline Settings
embedding_model: "all-MiniLM-L6-v2"
temporal_sigma: 300            # Seconds for temporal decay half-life

# Initial weights (Sum must be 1.0)
fusion_weights:
  temporal: 0.3
  semantic: 0.5
  topological: 0.2

# HDBSCAN clustering settings
hdbscan:
  min_cluster_size: 3
  min_samples: 2

# Root cause heuristic weights
root_cause_weights:
  time: 0.5
  severity: 0.3
  centrality: 0.2

avg_triage_minutes_per_alert: 2
```

---

# PHASE 1: DATA MODEL DEFINITIONS

## `models/alert.py`
```python
from pydantic import BaseModel, Field
from datetime import datetime

class Alert(BaseModel):
    id: str
    timestamp: datetime
    source: str
    severity: str                  # "critical" | "warning" | "info"
    service: str                   # mapped from BGL component (e.g., KERNEL, APP)
    host: str                      # BGL node coordinates (e.g., R02-M1-N0-C:J12-U11)
    message: str                   # log message content
    raw_label: str                 # original BGL first column tag (e.g., APPRECOVERY, -)
    ground_truth_cluster: str      # combined label + rack for accuracy validation
    tags: list[str] = Field(default_factory=list)
```

## `models/incident.py`
```python
from pydantic import BaseModel
from models.alert import Alert

class Incident(BaseModel):
    id: str
    alerts: list[Alert]
    alert_count: int
    root_cause: Alert
    summary: str
    confidence: float
    services_affected: list[str]
    severity_distribution: dict[str, int]
    engineer_hours_saved: float

class EvaluationMetrics(BaseModel):
    nmi_accuracy: float            # Normalized Mutual Information score
    ari_accuracy: float            # Adjusted Rand Index score
    silhouette_score: float        # Cluster separation score (-1 to 1)

class CorrelationResult(BaseModel):
    incidents: list[Incident]
    metrics: dict                  # general pipeline metrics
    evaluation: EvaluationMetrics
    noise_alerts: list[Alert]
```

---

# PHASE 2: DATA FETCHING & PARSING

## `data/downloader.py`
```python
"""Automated downloader for the Loghub BGL log dataset."""
import httpx
from pathlib import Path

BGL_2K_URL = "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log"
DATA_DIR = Path(__file__).parent.parent / "data_files"

def ensure_dataset_exists() -> Path:
    """Download BGL_2k.log from GitHub if it does not exist locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DATA_DIR / "BGL_2k.log"
    
    if not target_path.exists():
        print(f"Downloading BGL_2k.log from {BGL_2K_URL}...")
        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(BGL_2K_URL)
            resp.raise_for_status()
            target_path.write_bytes(resp.content)
        print("✅ Download complete")
    return target_path
```

## `data/bgl_parser.py`
```python
"""Parser for raw BGL logs with zero data loss."""
import re
import hashlib
from datetime import datetime
from pathlib import Path
from models.alert import Alert

# Regex matching BGL log columns
BGL_REGEX = re.compile(
    r'^(\S+)\s+'         # 1. Alert Label (e.g., APPRECOVERY or -)
    r'(\d+)\s+'          # 2. Epoch Timestamp
    r'([\d\.]+)\s+'      # 3. Date
    r'(\S+)\s+'          # 4. Node Coordinates
    r'([\d\.-]+)\s+'     # 5. Precise Timestamp
    r'(\S+)\s+'          # 6. Repeated Coordinates
    r'(\S+)\s+'          # 7. Component / Service
    r'(\S+)\s+'          # 8. Level
    r'(.*)$'             # 9. Content Message
)

SEVERITY_MAP = {
    "fatal": "critical", "severe": "critical", "error": "critical", "failure": "critical",
    "warning": "warning", "warn": "warning", "info": "info"
}

def parse_bgl_line(line: str) -> Alert | None:
    match = BGL_REGEX.match(line.strip())
    if not match:
        return None
        
    label = match.group(1)
    epoch = int(match.group(2))
    node = match.group(4)
    precise_ts_str = match.group(5)
    component = match.group(7)
    level = match.group(8).lower()
    msg = match.group(9)
    
    try:
        timestamp = datetime.strptime(precise_ts_str, "%Y-%m-%d-%H.%M.%S.%f")
    except ValueError:
        timestamp = datetime.fromtimestamp(epoch)
        
    # Map severity
    is_alert = label != "-"
    severity = SEVERITY_MAP.get(level, "critical" if is_alert else "info")
    
    # Generate deterministic alert ID
    line_hash = hashlib.md5(line.strip().encode()).hexdigest()[:12]
    
    # Build tags
    tags = [f"label:{label}", f"level:{level}"]
    rack = node.split("-")[0] if "-" in node else "unknown"
    tags.append(f"rack:{rack}")
    
    # Ground truth cluster logic:
    # If it is an alert, its ground truth group is the alert label combined with the rack.
    # If not an alert, it is classified as normal system noise ("noise").
    gt_cluster = f"{label}_{rack}" if is_alert else "noise"
    
    return Alert(
        id=f"bgl-{line_hash}",
        timestamp=timestamp,
        source="BlueGeneL-MMCS",
        severity=severity,
        service=component,
        host=node,
        message=msg,
        raw_label=label,
        ground_truth_cluster=gt_cluster,
        tags=tags
    )

def load_bgl_alerts(file_path: Path, alerts_only: bool = True) -> list[Alert]:
    alerts = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            alert = parse_bgl_line(line)
            if alert:
                if alerts_only and alert.raw_label == "-":
                    continue
                alerts.append(alert)
    return alerts
```

---

# PHASE 3: SCORING & FUSION

## `core/dedup.py`
```python
import re
import hashlib
from models.alert import Alert

def normalize_msg(msg: str) -> str:
    m = msg.strip().lower()
    m = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', m)
    m = re.sub(r'\d+', '<NUM>', m)
    return m

def deduplicate(alerts: list[Alert]) -> tuple[list[Alert], int]:
    seen = set()
    unique = []
    for a in alerts:
        msg_hash = hashlib.sha256(normalize_msg(a.message).encode()).hexdigest()
        if msg_hash not in seen:
            seen.add(msg_hash)
            unique.append(a)
    return unique, len(alerts) - len(unique)
```

## `core/embedder.py`
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
```

## `core/scorers.py`
```python
import math
import re
import numpy as np
from datetime import datetime
from models.alert import Alert

BGL_TOPOLOGY_REGEX = re.compile(r'^(R\d{2})-(M\d{1})-(N\d{2})')

def temporal_score(t1: datetime, t2: datetime, sigma: float = 300.0) -> float:
    delta = abs((t1 - t2).total_seconds())
    return math.exp(-(delta ** 2) / (2 * sigma ** 2))

def semantic_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))

def topological_score(alert1: Alert, alert2: Alert) -> float:
    """Computes layout similarity using hierarchical coordinates: Rack -> Midplane -> Card."""
    m1 = BGL_TOPOLOGY_REGEX.match(alert1.host)
    m2 = BGL_TOPOLOGY_REGEX.match(alert2.host)
    
    if m1 and m2:
        r1, mid1, n1 = m1.groups()
        r2, mid2, n2 = m2.groups()
        
        if r1 != r2:
            return 0.0     # Different racks
        if mid1 != mid2:
            return 0.4     # Same rack, different midplanes
        if n1 != n2:
            return 0.7     # Same midplane, different node cards
        return 1.0         # Same node card
        
    # Fallback to Jaccard similarity if host format is different
    s1 = {alert1.service, alert1.host} | set(alert1.tags)
    s2 = {alert2.service, alert2.host} | set(alert2.tags)
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0
```

## `core/fusion.py`
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.alert import Alert
from core.scorers import temporal_score, topological_score

def compute_fusion_matrix(alerts: list[Alert], embeddings: np.ndarray, weights: dict, sigma: float = 300.0) -> np.ndarray:
    n = len(alerts)
    w_t = weights["temporal"]
    w_s = weights["semantic"]
    w_p = weights["topological"]
    
    # Semantic cosine matrix
    sem_matrix = np.clip(cosine_similarity(embeddings), 0.0, 1.0)
    
    # Pairwise temporal & topological matrices
    temp_matrix = np.eye(n)
    topo_matrix = np.eye(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            t = temporal_score(alerts[i].timestamp, alerts[j].timestamp, sigma)
            p = topological_score(alerts[i], alerts[j])
            temp_matrix[i, j] = temp_matrix[j, i] = t
            topo_matrix[i, j] = topo_matrix[j, i] = p
            
    similarity = w_t * temp_matrix + w_s * sem_matrix + w_p * topo_matrix
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return np.clip(distance, 0.0, 1.0)
```

---

# PHASE 4: CLUSTERING & ROOT CAUSE

## `core/clusterer.py`
```python
import numpy as np
from sklearn.cluster import HDBSCAN

def cluster_alerts(distance_matrix: np.ndarray, min_cluster_size: int = 3, min_samples: int = 2) -> tuple[np.ndarray, np.ndarray]:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom"
    )
    clusterer.fit(distance_matrix)
    return clusterer.labels_, clusterer.probabilities_
```

## `core/root_cause.py`
```python
import numpy as np
from models.alert import Alert

SEV_MAP = {"critical": 1.0, "warning": 0.6, "info": 0.2}

def identify_root_cause(cluster: list[Alert], w_time=0.5, w_sev=0.3, w_cent=0.2) -> Alert:
    n = len(cluster)
    if n <= 1:
        return cluster[0]
        
    # Time scores (earliest = highest)
    sorted_idx = sorted(range(n), key=lambda i: cluster[i].timestamp)
    time_scores = [0.0] * n
    for rank, idx in enumerate(sorted_idx):
        time_scores[idx] = 1.0 - (rank / (n - 1)) if n > 1 else 1.0
        
    # Severity scores
    sev_scores = [SEV_MAP.get(a.severity, 0.2) for a in cluster]
    
    # Centrality (service density within cluster)
    svc_counts = {}
    for a in cluster:
        svc_counts[a.service] = svc_counts.get(a.service, 0) + 1
    max_svc = max(svc_counts.values())
    cent_scores = [svc_counts[a.service] / max_svc for a in cluster]
    
    composite = [
        w_time * time_scores[i] + w_sev * sev_scores[i] + w_cent * cent_scores[i]
        for i in range(n)
    ]
    return cluster[int(np.argmax(composite))]
```

## `core/summarizer.py`
```python
from models.alert import Alert

class Summarizer:
    def __init__(self, gemini_key: str = "", groq_key: str = ""):
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        
    def summarize(self, alerts: list[Alert], root: Alert) -> str:
        prompt = (
            f"Write exactly ONE short sentence summarizing this cluster of {len(alerts)} supercomputer alerts.\n"
            f"Primary failure: [{root.severity}] Node: {root.host} Component: {root.service} Message: {root.message}\n"
            f"Return only the summary. Do not include introductory text."
        )
        
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel("gemini-2.0-flash")
                return model.generate_content(prompt).text.strip()
            except Exception:
                pass
                
        if self.groq_key:
            try:
                from groq import Groq
                client = Groq(api_key=self.groq_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass
                
        # Default template
        services = sorted(list(set(a.service for a in alerts)))
        return f"{root.raw_label} event on {root.host} in component {root.service} caused alerts in {', '.join(services[:3])}."
```

---

# PHASE 5: EVALUATION & ACCURACY OPTIMIZATION

## `core/evaluator.py`
```python
"""Clustering quality evaluation metrics using scikit-learn."""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_samples
from models.alert import Alert
from models.incident import EvaluationMetrics

def evaluate_clustering(
    alerts: list[Alert],
    labels: np.ndarray,
    distance_matrix: np.ndarray
) -> EvaluationMetrics:
    """
    Evaluate predicted cluster labels against ground truth labels.
    
    Ground truth for BGL alerts: raw alert labels + rack location.
    If only one cluster is found or all alerts are categorized as noise, 
    metrics degrade gracefully (returns 0.0).
    """
    # Exclude noise alerts (-1) from supervised metrics comparison
    valid_idx = np.where(labels != -1)[0]
    
    if len(valid_idx) < 2:
        return EvaluationMetrics(nmi_accuracy=0.0, ari_accuracy=0.0, silhouette_score=0.0)
        
    gt_labels = [alerts[i].ground_truth_cluster for i in valid_idx]
    pred_labels = labels[valid_idx]
    
    # Calculate Supervised Metrics
    nmi = normalized_mutual_info_score(gt_labels, pred_labels)
    ari = adjusted_rand_score(gt_labels, pred_labels)
    
    # Calculate Unsupervised Metric (Silhouette) on valid distance matrix slice
    try:
        if len(set(pred_labels)) > 1:
            valid_dist = distance_matrix[valid_idx][:, valid_idx]
            # Silhouette on distance matrix: pass metric='precomputed'
            # Average silhouette score
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(valid_dist, pred_labels, metric="precomputed"))
        else:
            sil = 0.0
    except Exception:
        sil = 0.0
        
    return EvaluationMetrics(
        nmi_accuracy=round(float(nmi), 4),
        ari_accuracy=round(float(ari), 4),
        silhouette_score=round(sil, 4)
    )
```

## `core/optimizer.py`
```python
"""Weight optimization module to maximize clustering accuracy (NMI)."""
import numpy as np
from models.alert import Alert
from core.fusion import compute_fusion_matrix
from core.clusterer import cluster_alerts
from core.evaluator import evaluate_clustering

def optimize_weights(
    alerts: list[Alert],
    embeddings: np.ndarray,
    hdbscan_params: dict,
    sigma: float = 300.0,
    steps: int = 5
) -> tuple[dict, float]:
    """
    Performs grid search to find the optimal fusion weights (temporal, semantic, topological)
    that maximize the Normalized Mutual Information (NMI) score on the parsed dataset.
    
    Constraints: weights must sum to 1.0.
    """
    best_weights = {"temporal": 0.3, "semantic": 0.5, "topological": 0.2}
    best_nmi = 0.0
    
    # Generate weight candidates summing to 1.0
    grid = np.linspace(0.0, 1.0, steps + 1)
    
    for wt in grid:
        for ws in grid:
            wp = 1.0 - wt - ws
            if wp < -0.01:
                continue
            wp = max(0.0, wp)
            
            candidate_weights = {"temporal": wt, "semantic": ws, "topological": wp}
            
            # Compute fusion distance matrix
            dist = compute_fusion_matrix(alerts, embeddings, candidate_weights, sigma)
            
            # Run clustering
            labels, _ = cluster_alerts(dist, hdbscan_params["min_cluster_size"], hdbscan_params["min_samples"])
            
            # Evaluate NMI
            eval_metrics = evaluate_clustering(alerts, labels, dist)
            
            if eval_metrics.nmi_accuracy > best_nmi:
                best_nmi = eval_metrics.nmi_accuracy
                best_weights = candidate_weights
                
    return best_weights, best_nmi
```

---

# PHASE 6: ORCHESTRATION PIPELINE

## `core/engine.py`
```python
import time
import numpy as np
from models.alert import Alert
from models.incident import Incident, CorrelationResult, EvaluationMetrics
from core.dedup import deduplicate
from core.embedder import Embedder
from core.fusion import compute_fusion_matrix
from core.clusterer import cluster_alerts
from core.root_cause import identify_root_cause
from core.summarizer import Summarizer
from core.evaluator import evaluate_clustering

class CorrelationEngine:
    def __init__(self, config: dict):
        self.embedder = Embedder(config.get("embedding_model", "all-MiniLM-L6-v2"))
        self.summarizer = Summarizer(
            gemini_key=config.get("gemini_api_key", ""),
            groq_key=config.get("groq_api_key", "")
        )
        self.weights = config.get("fusion_weights", {"temporal": 0.3, "semantic": 0.5, "topological": 0.2})
        self.hdbscan_params = config.get("hdbscan", {"min_cluster_size": 3, "min_samples": 2})
        self.sigma = config.get("temporal_sigma", 300.0)
        self.rc_weights = config.get("root_cause_weights", {"time": 0.5, "severity": 0.3, "centrality": 0.2})
        self.triage_min = config.get("avg_triage_minutes_per_alert", 2)
        
    def correlate(self, alerts: list[Alert]) -> CorrelationResult:
        start_time = time.time()
        
        # 1. Deduplicate
        unique_alerts, dupes_removed = deduplicate(alerts)
        
        # 2. Embedding
        messages = [a.message for a in unique_alerts]
        embeddings = self.embedder.embed_batch(messages)
        
        # 3. Distance Matrix Fusion
        dist_matrix = compute_fusion_matrix(unique_alerts, embeddings, self.weights, self.sigma)
        
        # 4. Clustering
        labels, probs = cluster_alerts(dist_matrix, self.hdbscan_params["min_cluster_size"], self.hdbscan_params["min_samples"])
        
        # 5. Incident Construction
        incidents = []
        noise_alerts = []
        unique_labels = set(labels)
        
        for cid in sorted(unique_labels):
            mask = labels == cid
            indices = np.where(mask)[0]
            cluster_list = [unique_alerts[i] for i in indices]
            
            if cid == -1:
                noise_alerts = cluster_list
                continue
                
            root = identify_root_cause(
                cluster_list,
                self.rc_weights["time"],
                self.rc_weights["severity"],
                self.rc_weights["centrality"]
            )
            summary = self.summarizer.summarize(cluster_list, root)
            confidence = float(np.mean(probs[mask]))
            
            sev_dist = {}
            for a in cluster_list:
                sev_dist[a.severity] = sev_dist.get(a.severity, 0) + 1
                
            incidents.append(Incident(
                id=f"INC-{cid:03d}",
                alerts=cluster_list,
                alert_count=len(cluster_list),
                root_cause=root,
                summary=summary,
                confidence=round(confidence, 3),
                services_affected=sorted(list(set(a.service for a in cluster_list))),
                severity_distribution=sev_dist,
                engineer_hours_saved=round(len(cluster_list) * self.triage_min / 60, 1)
            ))
            
        # 6. Evaluation metrics calculation
        eval_metrics = evaluate_clustering(unique_alerts, labels, dist_matrix)
        
        elapsed = round(time.time() - start_time, 2)
        
        metrics = {
            "total_alerts_received": len(alerts),
            "duplicates_removed": dupes_removed,
            "unique_alerts": len(unique_alerts),
            "incidents_found": len(incidents),
            "noise_alerts": len(noise_alerts),
            "noise_reduction_pct": round((1 - len(incidents) / max(len(unique_alerts), 1)) * 100, 1),
            "processing_time_seconds": elapsed
        }
        
        return CorrelationResult(
            incidents=incidents,
            metrics=metrics,
            evaluation=eval_metrics,
            noise_alerts=noise_alerts
        )
```

---

# PHASE 7: FastAPI WEB API

## `api/routes.py`
```python
from fastapi import APIRouter, HTTPException
from models.alert import Alert
from models.incident import CorrelationResult
from core.optimizer import optimize_weights

router = APIRouter(prefix="/api", tags=["AIOps"])

def get_engine():
    from main import app
    return app.state.engine

@router.post("/alerts/correlate", response_model=CorrelationResult)
async def correlate_alerts(alerts: list[Alert]):
    if not alerts:
        raise HTTPException(status_code=400, detail="Alert list is empty")
    return get_engine().correlate(alerts)

@router.post("/config/optimize")
async def optimize_parameters(alerts: list[Alert]):
    """Optimize weights to yield maximum NMI clustering accuracy."""
    if not alerts:
        raise HTTPException(status_code=400, detail="Alert list is empty")
        
    engine = get_engine()
    unique_alerts, _ = engine.correlate(alerts) # quick run for deduplication
    
    # Re-extract uniques for optimization run
    from core.dedup import deduplicate
    uniques, _ = deduplicate(alerts)
    messages = [a.message for a in uniques]
    embeddings = engine.embedder.embed_batch(messages)
    
    best_weights, best_nmi = optimize_weights(
        uniques, embeddings, engine.hdbscan_params, engine.sigma, steps=6
    )
    
    engine.weights = best_weights
    return {"status": "optimized", "best_weights": best_weights, "max_nmi": round(best_nmi, 4)}

@router.put("/config/weights")
async def update_weights(weights: dict):
    req_keys = {"temporal", "semantic", "topological"}
    if not req_keys.issubset(weights.keys()):
        raise HTTPException(status_code=400, detail="Missing required weight labels")
    get_engine().weights = weights
    return {"status": "weights_updated", "weights": weights}
```

## `main.py`
```python
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from core.engine import CorrelationEngine

app = FastAPI(title="HPE Alert Correlation Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
async def startup():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    app.state.engine = CorrelationEngine(config)
    print("🚀 HPE BGL Alert Correlation Engine Started.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

# PHASE 8: DASHBOARD & METRICS DISPLAY

## `dashboard/app.py`
```python
"""Streamlit front-end for HPE Alert Correlation Engine."""
import streamlit as st
import httpx
import json
from pathlib import Path

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="HPE Alert Correlation Engine", layout="wide")
st.title("🚨 BlueGene/L Supercomputer Alert Correlation Engine")

# Load configuration values to show default values
st.sidebar.header("🎛️ Pipeline Weights")
wt = st.sidebar.slider("Temporal Weight", 0.0, 1.0, 0.3, 0.05)
ws = st.sidebar.slider("Semantic Weight", 0.0, 1.0, 0.5, 0.05)
wp = st.sidebar.slider("Topological Weight", 0.0, 1.0, 0.2, 0.05)

# Normalise weights
total_w = wt + ws + wp
if total_w > 0:
    wt /= total_w
    ws /= total_w
    wp /= total_w
    
st.sidebar.caption(f"Normalized: Time: {wt:.2f} | Text: {ws:.2f} | Space: {wp:.2f}")

# Fetching or loading original BGL log dataset
st.sidebar.header("📂 Dataset Control")

# Automatically triggers dataset downloader
from data.downloader import ensure_dataset_exists
from data.bgl_parser import load_bgl_alerts

try:
    bgl_path = ensure_dataset_exists()
    st.sidebar.success("BGL Dataset loaded locally")
except Exception as e:
    st.sidebar.error(f"Failed to fetch dataset: {e}")
    st.stop()

max_records = st.sidebar.number_input("Max BGL lines to load", 100, 2000, 1000, step=100)
alerts_only = st.sidebar.checkbox("Load alert log lines only", value=True)

# Parse file
alerts = load_bgl_alerts(bgl_path, alerts_only=alerts_only)
alerts = alerts[:max_records]

# Convert alerts to JSON lists for API payload
payload = [a.model_dump(mode="json") for a in alerts]

# --- Optimization Action ---
if st.sidebar.button("🤖 Run Weight Optimizer", use_container_width=True):
    with st.spinner("Finding optimal weights on BGL dataset..."):
        try:
            opt_resp = httpx.post(f"{API_URL}/config/optimize", json=payload, timeout=60)
            opt_data = opt_resp.json()
            best = opt_data["best_weights"]
            st.sidebar.success(f"Optimized! Max NMI: {opt_data['max_nmi']:.1%}")
            st.sidebar.info(f"Set: T={best['temporal']:.2f}, S={best['semantic']:.2f}, P={best['topological']:.2f}")
            # Refresh using optimized weights
            wt, ws, wp = best["temporal"], best["semantic"], best["topological"]
        except Exception as e:
            st.sidebar.error(f"Optimization failed: {e}")

# --- Core dashboard actions ---
col1, col2 = st.columns(2)

with col1:
    st.header(f"📥 Raw BGL Log Stream ({len(alerts)} records)")
    with st.expander("Show raw entries"):
        for a in alerts[:50]:
            st.text(f"[{a.severity.upper()}] Component: {a.service} | Node: {a.host} | Message: {a.message[:100]}")

# Process correlation
if st.button("⚡ Process & Cluster Alerts", use_container_width=True, type="primary"):
    # Send updated configuration weights to FastAPI
    try:
        httpx.put(f"{API_URL}/config/weights", json={"temporal": wt, "semantic": ws, "topological": wp})
    except Exception:
        pass
        
    with st.spinner("Executing fusion & density-based clustering pipeline..."):
        try:
            correlate_resp = httpx.post(f"{API_URL}/alerts/correlate", json=payload, timeout=60)
            res = correlate_resp.json()
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            st.stop()
            
    # Metrics
    metrics = res["metrics"]
    eval_m = res["evaluation"]
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Alerts Received", metrics["total_alerts_received"])
    m2.metric("Incident Clusters Found", metrics["incidents_found"])
    m3.metric("Noise Reduction Ratio", f"{metrics['noise_reduction_pct']}%")
    
    e1, e2, e3 = st.columns(3)
    e1.metric("NMI Clustering Accuracy", f"{eval_m['nmi_accuracy']:.1%}")
    e2.metric("Adjusted Rand Index (ARI)", f"{eval_m['ari_accuracy']:.1%}")
    e3.metric("Silhouette Separation Score", f"{eval_m['silhouette_score']:.2f}")
    
    with col2:
        st.header(f"🗂️ Correlated Incidents ({len(res['incidents'])})")
        for inc in res["incidents"]:
            with st.container(border=True):
                st.subheader(f"INCIDENT: {inc['id']}")
                st.write(f"**Root Cause Alert:** `[{inc['root_cause']['severity'].upper()}] {inc['root_cause']['service']}/{inc['root_cause']['host']}: {inc['root_cause']['message']}`")
                st.write(f"**LLM Digest:** *\"{inc['summary']}\"*")
                st.write(f"**Metrics:** Size: {inc['alert_count']} alerts | Confidence: {inc['confidence']:.1%} | Estimated Triage Savings: {inc['engineer_hours_saved']} hours")
                
                with st.expander("Expand nested alerts"):
                    for a in inc["alerts"]:
                        st.write(f"- `[{a['severity'].upper()}]` {a['message']}")
```

---

# PHASE 9: VERIFICATION PLAN

Run the system locally and complete the verification steps:

```bash
# 1. Start backend server
python main.py

# 2. Run streamlit dashboard (in separate window)
streamlit run dashboard/app.py
```

### Checks
1. Check that `data_files/BGL_2k.log` is downloaded automatically on dashboard boot.
2. Verify that clicking **Process & Cluster Alerts** generates incidents in column 2 and outputs non-zero NMI/ARI accuracy metrics.
3. Verify that running the **Weight Optimizer** finds new weights, prints the NMI score, and successfully updates the clustering outputs.
4. Verify that Gemini summaries degrade gracefully to the template summaries if the Gemini API key is left empty.
