# Automated Cluster Interpretation with a Multi-Agent Pipeline

> **The hard part of clustering is not the math — it's the meaning.**

---

## The Problem

Clustering appears in almost every domain of applied data science: segmenting customers by behaviour, grouping patients by symptom profile, categorising documents by topic, organising images by visual similarity, partitioning sensor readings by operational mode. The algorithms are well established — k-means, hierarchical, GMM, DBSCAN, etc. — and any of them can produce as many clusters as you ask for in seconds.

The unsolved part is what comes after. What do those clusters *mean*? What should each one be called? What makes one group different from its neighbours — not in terms of centroid coordinates, but in terms a business or scientific audience can act on? Clustering is an unsupervised task: there are no ground-truth labels, so you cannot measure accuracy. The challenge is not prediction. It is **interpretability**.

Without automation, this loop typically runs multiple times, each iteration requiring the labelling step to be redone in full. The result is a process that is slow (days per project), undocumented (diagnostic reasoning is rarely recorded), and non-reproducible (the same data produces different segments depending on who runs the analysis).

The system described here automates the entire loop — feature engineering, selection, clustering, constraint checking, contrastive labelling, and iterative diagnosis — using a **multi-agent architecture** in which a Decision Maker (any LLM API) handles the steps that require judgment. The result: a complete, named, validated cluster solution in under one hour, at under one dollar of API cost, with a full reasoning trace for every decision.

---

## The Agent Approach

The pipeline is driven by **`run_pipeline.py`**. Seven specialised agents plus a Decision Maker form a feedback loop. Every quality gate can push the pipeline backward; it only moves forward when all gates pass (or the user approves):

<img width="1097" height="592" alt="image" src="https://github.com/user-attachments/assets/97aa473b-0055-452b-bb87-448cb1d701fb" />


### What each agent does

| # | Agent | Role |
|---|-------|------|
| ⓪ | **UserInputAgent** | Prompts for clustering intent (target entity, business purpose, dataset path, optional `modality` / `text_column`). |
| ① | **DatasetExaminerAgent** | Profiles the raw data — schema, missingness, distribution shape, cardinality — and asks the Decision Maker to suggest feature engineering groups aligned with the business purpose. Also emits an algorithm hint based on skewness. **Detects text-dominant datasets** (object/string column with high mean token count + high uniqueness) and routes the pipeline through the text branch. |
| ② | **FeatureEngineerAgent** *(tabular)* | Builds an entity-level feature matrix from raw event-level data. The Decision Maker reads the actual column names from the dataset schema and plans which of 8 generic statistical operations to apply (group aggregation, trends, streaks, diversity, temporal patterns, etc.). No domain vocabulary is hard-coded — the LLM reasons from the data. Saves to `data/processed/`. Skipped when a pre-built parquet is provided. |
| ② | **TextPreparerAgent** *(text)* | Replaces FeatureEngineer when the dataset is text-dominant. Recommends an embedding method via `text_vectorizer` (TF-IDF + TruncatedSVD for short text, sentence-transformers for long prose), vectorizes the documents, and saves an embedding parquet. Stashes the raw docs + TF-IDF vocab so the downstream Clusterer can build per-cluster c-TF-IDF distinctive terms and representative documents. |
| ③ | **FeatureSelectionAgent** | Scores all features with PCA importance and autoencoder reconstruction error, runs a VIF collinearity gate, then asks the Decision Maker to pick the best subset (typically 25–55 features). The VIF threshold and a feature-focus hint are set dynamically by the Decision Maker each iteration. **Short-circuits in text mode** — embedding dims are already compact and decorrelated, so PCA/AE/VIF are skipped and every dim is kept. |
| ④ | **ClusteringAgent** | Auto-selects the best algorithm from five options (`kmeans`, `hierarchical`, `dbscan`, `gmm`, `fuzzy_cmeans`) via `algo_recommender`. Auto-selects k via silhouette score optimisation. Runs a deepening loop to split any oversized cluster (>40%). All numeric columns are log-transformed automatically if skewed (|skewness| > 2.0). **In text mode** the matrix is L2-normalized (spherical-k-means semantics), silhouette is computed with cosine distance, and per-cluster profiles are built from **c-TF-IDF distinctive terms + centroid-nearest representative documents** instead of numeric means. |
| ⑤ | **PersonaNamingAgent** | Sends cluster profiles to the Decision Maker as tables of feature deviations from the global mean. The Decision Maker writes name, tagline, description, and five traits per cluster. A **Clarity Gate** (avg confidence ≥ 6.0, all names unique) must pass or the pipeline re-clusters. **In text mode** the prompt block shows `DISTINCTIVE TERMS` + `REPRESENTATIVE DOCUMENTS` instead of numeric mean deviations — same output schema, so the UI is unchanged. |
| ⑥ | **ClassifierAgent** | Asks the Decision Maker to select the best classifier (`random_forest`, `xgboost`, `gradient_boosting`, `logistic_regression`) for the data. Trains with stratified 5-fold CV. If macro-F1 < 0.70 (0.60 in text mode), the Decision Maker diagnoses the root cause and routes back to ③ or ④. |

### How each agent calls the Decision Maker

Every agent follows the same four-step pattern:

1. **Compute** — run sklearn / numpy / pandas to produce statistics (PCA scores, cluster profiles, silhouette scores, etc.).
2. **Format** — assemble those statistics into a structured text prompt in Python using actual column names discovered from the data.
3. **Call** — send the prompt through `OrchestratorBus` to the LLM API (any chat-completion endpoint — Claude, GPT, Gemini, etc.).
4. **Parse** — read the Decision Maker's JSON response and act on it.

The agents are Python scripts that construct precise, data-rich prompts and parse structured responses. All LLM access of agents is mediated by the Orchestrator through `OrchestratorBus`. The cluster statistics are computed at runtime from the actual feature matrix and injected into the prompt. The Decision Maker reads those numbers and returns structured JSON with `name`, `tagline`, `description`, `dominant_features`, `traits`, `confidence`.

**Concrete example — `PersonaNamingAgent`**

The function `build_all_clusters_prompt()` dynamically assembles a prompt using features discovered from the actual data:

```
You are a behavioral analyst interpreting entity clusters.
Each cluster is described by its most distinguishing features:
  vs_avg: ratio of cluster mean to overall mean (◀ = 40%+ above; ◀◀ = 100%+ above; ▼ = 50%+ below)
  mean: the cluster's average value for that feature

CLUSTER 0  (1 234 entities, 12.3% of all entities)
Algorithm: kmeans

  ABOVE AVERAGE (strongest signals):
    count_category_food_12m                   mean=      87.4  vs_avg=2.41x ◀◀
    sum_category_travel_12m                   mean=   8200.1  vs_avg=3.18x ◀◀
    streak_category_grocery_pos               mean=       9.1  vs_avg=1.72x ◀
    …

  BELOW AVERAGE:
    event_count_all_6m                        mean=      12.3  vs_avg=0.38x ▼
    …

CRITICAL NAMING RULES — read carefully before writing any name:
1. SPECIFICITY — names must describe what the entity ACTUALLY DOES …
…

Return ONLY a valid JSON object …
```
---

## Best-Effort Fallback

If 10 iterations complete without any result passing all gates, the pipeline does **not** just exit empty-handed. Instead it:

1. Identifies the iteration with the highest silhouette score across all attempts.
2. Runs **PersonaNamer** on that clustering with `force_proceed=True` (Clarity Gate bypassed).
3. Runs **Classifier** on the result.
4. Saves all outputs and returns `status='best_effort'`.

The console prints a `⚠ BEST-EFFORT RESULT` banner so the output is clearly flagged. This guarantees a full analysis is always delivered regardless of data difficulty.

---

## Text Modality (document / article clustering)

The same pipeline clusters **text** (articles, reviews, support tickets, posts) by routing it through the `TextPreparerAgent` instead of `FeatureEngineerAgent`. Every other stage (FeatureSelector, Clusterer, PersonaNamer, Classifier) stays the same — they just see embedding columns instead of numeric features.

### How routing works

`DatasetExaminerAgent` auto-detects text-dominant datasets (an object/string column whose mean token count is high and values are mostly unique). You can also force the routing:

```bash
# Auto-detect (recommended)
python run_pipeline.py --data data/raw/twenty_newsgroups/twenty_newsgroups.csv

# Force text modality + name the column explicitly
python run_pipeline.py \
  --data data/raw/twenty_newsgroups/twenty_newsgroups.csv \
  --modality text \
  --text-column text

# Equivalent via config.yaml:
#   modality: text
#   text_column: text
#   text_vectorizer: auto         # or tfidf_svd / transformer
```

### What changes when modality is `text`

| Stage | Behaviour |
|-------|-----------|
| `DatasetExaminerAgent` | Skips the "no numeric columns" hard-block; profiles the text column instead. |
| `TextPreparerAgent` | New. Picks an embedding method (`tfidf_svd` default; `transformer` if `sentence-transformers` is installed). Saves embeddings to `data/processed/text_embeddings.parquet`. |
| `FeatureSelectionAgent` | Short-circuits — PCA/AE/VIF don't apply to embeddings. Keeps all dims. |
| `ClusteringAgent` | L2-normalizes embeddings, uses **cosine silhouette**, builds **c-TF-IDF distinctive terms + representative docs** for each cluster. |
| `PersonaNamingAgent` | Prompt shows distinctive terms + doc snippets so the LLM names the *topic*, not a numeric segment. |
| `Orchestrator` | `min_silhouette` relaxes to **0.01** (cosine silhouettes are smaller than euclidean) and the classifier F1 gate to **0.60**. The failure-tuning LLM can swap `text_vectorizer` (tfidf_svd ↔ transformer) when iterations miss — that's the text-mode analog of "reselect features". |

### Public benchmark — 20 Newsgroups

A real, public, well-vetted text-clustering dataset (no Kaggle login, no scraping):

```bash
# One-time: download the corpus via scikit-learn into data/raw/twenty_newsgroups/
python data/raw/twenty_newsgroups/download.py

# Offline benchmark (no LLM calls — stubbed bus)
python experiments/benchmark_text_clustering.py

# Full pipeline via Orchestrator + stubbed LLM
python experiments/test_text_e2e_orchestrator.py
```

Expected output on the offline benchmark (1000 posts × 5 categories, TF-IDF + SVD):

```
[3] ClusteringAgent ... 5 clusters · cosine silhouette = 0.05
[5] Cluster purity   = 0.74  (random baseline 0.20 — 3.7× above chance)
[6] Cluster 0 terms: car (51.7) · cars (24.4) · dealer (16.5) · engine (16.0)
                     · speed (11.8) · tires (10.7) · ford (10.1) → rec.autos
```

---

## Data

The pipeline is dataset-agnostic. Point it at any tabular CSV where rows are events and columns include an entity identifier, a timestamp, and descriptive attributes, **or** at any CSV with a free-text column for document clustering. The `UserInputAgent` (and the `DatasetExaminerAgent`'s modality detection) will route into the right pipeline branch automatically.

### Demo dataset

The included demo uses the [**Fraud Detection**](https://www.kaggle.com/datasets/kartik2112/fraud-detection) dataset by Kartik Shenoy on Kaggle (`kartik2112/fraud-detection`). It contains ~1.3 million simulated credit-card transactions for ~983 cardholders, with columns for merchant, category, amount, timestamp, and demographics.

**Download** (requires a [Kaggle API token](https://www.kaggle.com/docs/api)):

```bash
pip install kaggle
kaggle datasets download -d kartik2112/fraud-detection -p data/raw --unzip
```

The pipeline uses `data/raw/fraudTrain.csv` (~335 MB) for feature engineering and clustering.

---

## How to Run

**Prerequisites**

```bash
pip install -r requirements.txt
export LLM_API_KEY="sk-ant-..."   # or add to .env
```

**Run from a raw event-level CSV**

```bash
python run_pipeline.py
# UserInputAgent will prompt for: entity being clustered, business purpose, dataset path
# FeatureEngineerAgent builds the feature matrix automatically
```

**Run from a pre-built feature parquet**

```bash
# If a feature parquet already exists, the pipeline skips FeatureEngineerAgent
# and reads it directly. No extra steps needed.
python run_pipeline.py
```

The script:

- Loads `.env` and `config.yaml`
- Auto-detects whether to run FeatureEngineerAgent (raw CSV) or skip it (parquet)
- Runs the Decision Maker loop with `max_total_iterations=10`
- After each failure, the Decision Maker proposes new VIF/k/algorithm/silhouette parameters
- At max iterations, delivers a best-effort result if no iteration fully passed
- Writes all outputs under `outputs/` and prints a full console report

---

## Configuration (`config.yaml`)

```yaml
# ── Clustering ──────────────────────────────────────────────────────
n_clusters: ~               # null = auto-select k via silhouette optimizer (recommended)
                            # Set an integer (e.g. 6) only to force a specific k

clustering_algorithm: auto  # auto | kmeans | hierarchical | dbscan | gmm | fuzzy_cmeans
                            # auto = AlgoRecommender scores all five and picks the best
                            # The Decision Maker may override this per-iteration

# ── Classifier ──────────────────────────────────────────────────────
classifier_model: auto      # auto | random_forest | xgboost | gradient_boosting | logistic_regression
                            # auto = Decision Maker selects based on data characteristics

# ── Deepening loop ──────────────────────────────────────────────────
max_cluster_size_pct: 0.40  # split any cluster larger than this share of total entities
sub_n_clusters: 3           # how many sub-clusters to create when splitting
max_depth: 2                # max splitting rounds (0 = disabled)

# ── Persona tone ────────────────────────────────────────────────────
persona_tone: easy          # easy | professional | data-driven | creative
```

**`n_clusters: ~` (null) is the default and recommended setting.** It lets the silhouette optimizer scan `[3, 4, 5, 6, 7, 8, 10, 12, 15]` and pick the best k automatically. Set an integer only when you have a specific business requirement.

**`clustering_algorithm: auto` is recommended.** The `AlgoRecommender` skill scores all five algorithms against data shape metrics (n_entities, n_features, skewness, outlier spread) and business purpose keywords, then picks the best fit. The Decision Maker can override after each iteration.

---

## Outputs

After a successful (or best-effort) run:

| File | Description |
|------|-------------|
| `outputs/personas.json` | Machine-readable personas: name, tagline, traits, cluster stats, lineage. |
| `outputs/persona_summary.txt` | Human-readable persona cards with top distinguishing features. |
| `outputs/persona_metrics.csv` | One row per cluster × distinguishing feature: `mean_value`, `relative_to_avg`, signal strength. |
| `outputs/classifier_metrics.json` | CV accuracy/F1, per-class F1, top feature importances, reasoning. |
| `outputs/cluster_profiles.json` | Raw per-cluster statistics: `n_entities`, `pct_total`, `top_above_average`, `top_below_average`, `feature_means`. |
| `outputs/cluster_lineage.json` | Cluster tree: parent/child relationships from the deepening loop. |
| `outputs/silhouette_curve.json` | k vs silhouette score curve from the optimizer, best k, algorithm reasoning. |
| `outputs/pipeline_log.json` | Full structured log of every agent's status report across all iterations. |
| `outputs/agents_conversation.txt` | Full text log of every LLM prompt and response. |
| `data/processed/engineered_features.parquet` | Entity-level feature matrix built by FeatureEngineerAgent (when starting from CSV). |
| `data/processed/text_embeddings.parquet` | Document × embedding-dim matrix produced by TextPreparerAgent in text mode (one row per document, columns `emb_0…emb_n`). |

---

## Skills

The agents do not have hard-coded logic for every decision. They call shared **skills** — focused Python modules — for statistical tasks, and route Decision Maker queries through `OrchestratorBus`:

| Skill | File | Used by |
|-------|------|---------|
| **OrchestratorBus** | `skills/orchestrator_bus.py` | All agents — the sole LLM gateway; logs every prompt and response |
| **VIF checker** | `skills/vif_checker.py` | FeatureSelector — multicollinearity gate |
| **Silhouette optimizer** | `skills/silhouette_optimizer.py` | Clusterer — auto k-selection (supports `metric='cosine'` for text mode) |
| **Algorithm recommender** | `skills/algo_recommender.py` | Clusterer — scores 5 algorithms and recommends the best fit |
| **Text vectorizer** | `skills/text_vectorizer.py` | TextPreparer — recommends an embedding method (TF-IDF + SVD or sentence-transformer) and vectorizes documents, falling back to TF-IDF when `sentence-transformers` is unavailable |

---

## Setup (Quick)

```bash
pip install -r requirements.txt
export LLM_API_KEY="..."   # or add to .env
python run_pipeline.py
```

Open **`outputs/persona_summary.txt`** for full persona cards and **`outputs/persona_metrics.csv`** for structured metrics after the run.
