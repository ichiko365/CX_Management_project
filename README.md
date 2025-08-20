## CX Management Project

Customer Experience (CX) Management project for collecting product reviews, storing them via a FastAPI backend, running an analysis pipeline, and exploring insights in a dashboard. The project ships with a Customer App (review intake), a Client Dashboard (KPIs/insights), and a single Streamlit Hub to start and navigate everything.

Tip: The easiest way to start is the Hub: run DashBoard_streamlit.py and use its buttons/links to open apps.

### Key components
- Customer App: `app/app.py`
	- Streamlit UI to submit reviews (ASIN + Review required; Title/Description optional and can auto-fill from DB).
	- Auto-starts the FastAPI backend on the first run and shows basic health/port info.
	- Can trigger the pipeline and tails logs.
- FastAPI Backend: `app/main.py` (and `app/main_temp.py`)
	- Endpoints like POST `/add_review/` and POST `/trigger_pipeline`.
	- Inserts reviews and launches `run_pipeline.py` as needed.
- Pipeline: `run_pipeline.py`
	- Processes reviews (e.g., sentiment/LLM analysis) and writes results/logs.
- Client Dashboard: `src/dashboard_project/app.py`
	- Streamlit dashboard for KPIs and insights.
- Streamlit Hub (recommended entry): `DashBoard_streamlit.py`
	- Starts/embeds the Customer App and Client Dashboard, shows status, and offers quick links.

---

## Requirements
- macOS, Linux, or Windows
- Python 3.10+ recommended
- pip or uv
- Database (e.g., PostgreSQL) if you plan to persist data; set `DATABASE_URL` accordingly

Optional but helpful:
- A `.env` file at the repo root for secrets and connection strings
- Streamlit extras for nicer UI (optional)

---

## Setup
1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # zsh/bash on macOS/Linux
```

2) Install dependencies

Install through requirements.txt if present:

```bash
pip install -r requirements.txt
```
Then, download Ollama and follow the installation instructions.
> Then, download the model weights for your chosen LLM (e.g., "deepseek-r1:8b") and place them in the appropriate directory.
to download `deepseek-r1:8b`, you can use the following command:
```bash
ollama pull deepseek-r1:8b
```
Finally, Download PostgreSql

3) Configure environment

### Create .env (format)
```env
DB_USER="postgres"
DB_PASSWORD=""
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME=""
# "cx_hackathon_db"

# Results Database
RESULT_DB_NAME=""
# "cx_hackathon_results_db"
```

### For Streamlit
You can also place sensitive values in Streamlit secrets if you prefer: `app/.streamlit/secrets.toml` and `src/dashboard_project/.streamlit/secrets.toml`.

4) Run `run_pipeline.py` to start the review processing pipeline.

Here, you can monitor the processing of reviews and any potential issues, and put into the results database using LLMs.
Here, you can change:
        the batch size for processing reviews: line 30
        the model name for LLM analysis: line 47

5) Final
    - Now, everything is ready for the first run!
    - You can `DashBoard_streamlit.py` to start the hub and navigate to the apps.


## Run apps directly (optional)

Customer App (auto-starts backend on first run):
```bash
streamlit run app/app.py
```

Client Dashboard:
```bash
streamlit run src/dashboard_project/app.py
```

Backend API only (usually not needed because the Customer App starts it):
```bash
python -m uvicorn app.main:app --reload
```

Pipeline (can also be triggered from the Customer App UI):
```bash
python run_pipeline.py
```

---

## Using the Customer App
- Fill in ASIN and Review (required). Title/Description are optional; if omitted, the app may auto-fill from the DB using ASIN.
- Submit the form to insert into the database via the FastAPI backend.
- The app batches pipeline runs (e.g., after every N reviews) and also provides a “Run Pipeline Now” button.
- Check logs in `logs/` (e.g., `backend.log`, `pipeline.log`) using the expanders in the UI.

---

## Project structure (abridged)

```
CX_Management_project/
	app/
		app.py             # Customer App (Streamlit)
		main.py            # FastAPI backend (auto-started by app/app.py)
		schema.py, db_engine.py
	src/
		dashboard_project/
			app.py           # Client Dashboard (Streamlit)
	run_pipeline.py      # Review processing pipeline
	DashBoard_streamlit.py # Hub to start/preview both apps (recommended entry)
	logs/
	.env (optional)
```

---

## Recommended entry point

Use the Streamlit Hub last to navigate “all over the place” (start apps, open links, embed views):

```bash
streamlit run DashBoard_streamlit.py
```
