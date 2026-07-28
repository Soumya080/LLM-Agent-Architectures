# Local LLM Master Generation Prompt

Use the prompt below to direct your local model (such as `qwen2.5-coder:7b` or `llama3`) to implement the entire codebase.

Because local models can truncate output if asked to write too much at once, the prompt is structured as a **staged conversation flow**. Copy and paste these prompts one step at a time into your local model's interface.

---

### STEP 1: Feed the Specification to the Model
**Paste this prompt first to establish the context:**

```text
You are a Staff AI Engineer building an enterprise-grade Alert Correlation & Deduplication Engine for the Synergy 2026 HPE Hackathon. 

Our goal is to process the original Loghub BGL (BlueGene/L) supercomputer log dataset, cluster alerts, identify root causes, calculate accuracy metrics (NMI, ARI, Silhouette), and tune parameters using a grid-search optimizer.

Below is the directory structure we need to create. Acknowledge this structure, then wait for my command to generate each file. Do not write any code yet.

alert_engine/
├── main.py
├── config.yaml
├── requirements.txt
│
├── models/
│   ├── __init__.py
│   ├── alert.py
│   └── incident.py
│
├── core/
│   ├── __init__.py
│   ├── dedup.py
│   ├── embedder.py
│   ├── scorers.py
│   ├── fusion.py
│   ├── clusterer.py
│   ├── root_cause.py
│   ├── summarizer.py
│   ├── evaluator.py
│   ├── optimizer.py
│   └── engine.py
│
├── data/
│   ├── __init__.py
│   ├── downloader.py
│   └── bgl_parser.py
│
├── api/
│   ├── __init__.py
│   └── routes.py
│
└── dashboard/
    └── app.py
```

---

### STEP 2: Generate Config, Requirements, and Data Models
**Once the model acknowledges, paste this prompt:**

```text
Excellent. Now, generate the following foundation files. 
Provide the complete code for each file. Do not use placeholders, comments representing missing code, or 'TODOs'.

1. alert_engine/requirements.txt
2. alert_engine/config.yaml
3. alert_engine/models/alert.py
4. alert_engine/models/incident.py

Ensure the alert model parses BGL host strings, maps severity levels, and tracks ground-truth groups. Write the full files now.
```

---

### STEP 3: Generate the Data Downloader and Parser
**Once Step 2 is completed, paste this prompt:**

```text
Now generate the data acquisition and parsing layer. 
Provide the complete code for:

1. alert_engine/data/downloader.py (which pulls BGL_2k.log from GitHub if not present locally)
2. alert_engine/data/bgl_parser.py (which uses a robust regex to parse all 9 BGL columns, mapping severities, timestamps, components, and generating ground truth groupings for evaluation)

Do not abbreviate any code. Write the full implementations now.
```

---

### STEP 4: Generate Deduplication, Embeddings, and Scorers
**Paste this prompt next:**

```text
Generate the core analytical components. 
Provide the complete, unabbreviated code for:

1. alert_engine/core/dedup.py (exact-hash message deduplication after normalization)
2. alert_engine/core/embedder.py (sentence-transformers all-MiniLM-L6-v2 wrapper)
3. alert_engine/core/scorers.py (implementing temporal decay, semantic similarity, and BlueGene physical topology distance calculated from Rxx-Mxx-Nxx coordinates)

Write the full files now.
```

---

### STEP 5: Generate Fusion, Clustering, and Root Cause Heuristics
**Paste this prompt next:**

```text
Generate the grouping pipeline files. 
Provide the complete code for:

1. alert_engine/core/fusion.py (fusing temporal, semantic, and physical topological scores into an n x n distance matrix)
2. alert_engine/core/clusterer.py (HDBSCAN precomputed pre-fit clustering)
3. alert_engine/core/root_cause.py (composite heuristic using earliest alert + severity + node centrality)

Ensure all math functions are fully implemented using NumPy. Write the full files now.
```

---

### STEP 6: Generate Evaluator, Optimizer, Summarizer, and Engine
**Paste this prompt next:**

```text
Generate the core AI engine orchestrator and metrics validation files. 
Provide the complete code for:

1. alert_engine/core/evaluator.py (calculates NMI, ARI, and Silhouette scores using scikit-learn)
2. alert_engine/core/optimizer.py (runs grid search to find the weights that maximize BGL NMI accuracy)
3. alert_engine/core/summarizer.py (Gemini/Groq call with structured fallback)
4. alert_engine/core/engine.py (CorrelationEngine class tying the whole pipeline together)

Write the full files now.
```

---

### STEP 7: Generate API and Frontend Dashboard
**Paste this prompt next:**

```text
Generate the interface layers. 
Provide the complete, copy-paste-ready code for:

1. alert_engine/api/routes.py (FastAPI correlation and weight optimization endpoints)
2. alert_engine/main.py (FastAPI app startup and engine mounting)
3. alert_engine/dashboard/app.py (Streamlit UI with weight sliders, BGL raw log expander, metrics cards for accuracy/NMI/ARI/Silhouette, and a trigger for the weight optimizer)

Ensure the Streamlit dashboard connects cleanly to the local FastAPI port. Write the full files now.
```
