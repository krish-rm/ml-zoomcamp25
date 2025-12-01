## Market Profile ML Midterm – Product Requirements

### Overview
Deliver a production-ready system that analyzes hourly historical candles for a single liquid asset (baseline `BTC-USD`) to derive Market Profile features and predict whether the next session will break above the current Value Area High (VAH). The solution must satisfy ML Zoomcamp midterm deliverables: complete README, notebook, `train.py`, `predict.py`, FastAPI service, Docker image, and reproducibility expectations.[^1]

### Goals
- Provide daily Market Profile analytics (POC, VAH, VAL, balance state) for the selected asset.
- Train multiple supervised models to forecast a binary outcome: “next session closes above current VAH by threshold τ”.
- Serve real-time scoring via REST API and document end-to-end reproducibility.
- Package project artifacts and instructions aligned with course evaluation criteria.[^1]

### Non-Goals
- Multi-asset coverage.
- Real-time streaming ingestion or Level 2 order-book features.
- Automated trade execution.
- Reinforcement learning or sentiment integration (captured as future work).

### Personas
- **Quant Researcher (primary):** Wants reproducible analytics and predictive insights for short-term strategy iteration.
- **Trading Engineer (secondary):** Needs an API to integrate predictions into downstream tools.
- **Peer Reviewer:** Must be able to clone repo, rebuild environment, and validate functionality quickly.[^1]

### Functional Requirements
1. **Data Ingestion**
   - Fetch 18–24 months of hourly OHLCV data using `yfinance` or equivalent public API.
   - Persist snapshots under `data/raw/` and publish cleaned parquet to `data/processed/`.
   - Provide `data/README.md` covering source, refresh cadence, schema.

2. **Market Profile Engine**
   - Partition daily sessions into 30-minute Time Price Opportunities.
   - Compute POC, VAH, VAL, volume-weighted metrics, balance/imbalance flags, and rolling statistics (e.g., VA range width, session volume delta).
   - Package logic as reusable module `src/features/market_profile.py`.

3. **Label Construction**
   - Define binary label `breaks_above_vah`: 1 if next day’s high exceeds current VAH by >= τ (configurable, default 0.5%).
   - Optionally derive secondary label `breaks_below_val` for future roadmap, but keep current scope binary.

4. **Exploratory Analysis**
   - Notebook section for descriptive stats, missing-value checks, feature distributions, target incidence, and correlation heatmaps.
   - Include at least three visualizations: Market Profile plot, session balance timeline, model feature importance bar chart.

5. **Model Training**
   - Baseline logistic regression with scaled continuous features.
   - Tree-based model (RandomForest or XGBoost) with hyperparameter tuning using cross-validation.
   - Optional temporal model (e.g., LightGBM with lagged features) if time permits; document comparison table of metrics (precision, recall, ROC AUC).
   - Save best model and preprocessing pipeline as serialized artifacts under `models/`.

6. **Evaluation & Reporting**
   - Use stratified train/validation/test split at session level.
   - Report ROC AUC, Precision@k, confusion matrix on holdout.
   - Record experiment summary in README and notebook.

7. **Scripts**
   - `train.py`: CLI entrypoint (`python train.py --config configs/train.yaml`) performing data load, feature build, train, evaluation, artifact export, and metrics JSON.
   - `predict.py`: Loads artifacts, accepts CSV path or serialized JSON input, outputs predictions to stdout/file.
   - `serve.py`: FastAPI app exposing `GET /health` and `POST /predict` accepting current session features; returns probability and binary decision.

8. **Configuration**
   - Store defaults in `configs/train.yaml` (data paths, label threshold, model hyperparameters).
   - Provide `configs/api_schema.json` describing required payload fields for API.

9. **Deployment & Infrastructure**
   - Dockerfile based on `python:3.11-slim`, install requirements, run `uvicorn serve:app --host 0.0.0.0 --port 9696`.
   - Document Docker build/run commands and example API request in README.
   - (Optional) Compose file to run API + mock data generator for local testing.

10. **Testing & Validation**
    - Unit tests for feature engineering (`tests/test_market_profile.py`) covering POC/VA calculations.
    - Smoke test for API using pytest + `TestClient`.
    - CI suggestion (GitHub Actions) to lint, run tests, and ensure notebook executes (optional but recommended).

### Data Requirements
- **Source:** Yahoo Finance hourly OHLCV for chosen ticker via `yfinance`.
- **Window:** Minimum 18 months to capture diverse regimes.
- **Schema:** `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- **Storage:** CSV snapshots (`data/raw/YYYYMMDD.csv`), parquet for processed features (`data/processed/market_profile.parquet`).
- **Versioning:** Document retrieval date and ensure deterministic preprocessing.

### Modeling Requirements
- Feature list: session POC/VAH/VAL, VA range width, balance flag, volume imbalance, rolling return metrics (1-day, 3-day), volatility (ATR), RSI or Stochastic optional.
- Preprocessing: scaling with `StandardScaler` or `RobustScaler`, categorical encoding (if any).
- Hyperparameter tuning: grid or randomized search; store best params.
- Model persistence: `artifacts/model.pkl`, `artifacts/preprocessor.pkl`, config metadata JSON.

### Evaluation & Success Metrics
- Primary: ROC AUC > 0.60, precision for positive breakout class > baseline (class frequency).
- Secondary: API latency < 200 ms for single prediction on Dockerized service (local test).
- Documentation completeness per course rubric.[^1]

### System Architecture
1. Offline pipeline (scheduler optional) orchestrating data fetch → feature build → training.
2. Artifact store (local `models/`) consumed by inference paths.
3. REST API container serving predictions for caller-supplied session metrics.
4. Notebook for analysis; scripts for reproducibility; Docker for deployment.

### Deliverables
- `README.md` with problem description, dataset instructions, run guides, API usage, Docker steps, evaluation summary, future work.
- `notebook.ipynb` with EDA/modeling narrative.
- `train.py`, `predict.py`, `serve.py`, optional `utils/`.
- `requirements.txt`, `Dockerfile`, `configs/`.
- `tests/` folder with coverage for features/API.
- Sample input JSON under `examples/` and expected output.
- `Makefile` or scripts to streamline commands (optional).

### Timeline & Milestones
1. **Week 1**
   - Data extraction & profiling.
   - Implement Market Profile engine.
   - Initial notebook EDA.

2. **Week 2**
   - Model training + evaluation.
   - Refactor to scripts & configs.
   - Implement FastAPI service.

3. **Week 3**
   - Dockerize, add tests, finalize documentation.
   - Dry run from clean clone.
   - Record demo (optional) and prepare submission materials.

### Risks & Mitigations
- **Data gaps:** Yahoo API throttling → cache local snapshots; add retry logic.
- **Class imbalance:** Use stratified splits, evaluate balancing techniques (class weights).
- **Feature correctness:** Validate Market Profile outputs against manual checks, add unit tests.
- **Peer review reproducibility:** Include hashed artifact versions, requirements, and explicit commands per rubric.[^1]

### Future Enhancements
- Multi-asset extension with configurable tickers.
- Real-time ingestion via websockets.
- L2/order-book derived features.
- Sentiment integration and reinforcement learning agent.

[^1]: [ML Zoomcamp project deliverables and evaluation criteria](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/projects/README.md)