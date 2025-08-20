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

If you have a consolidated requirements file, install from it. Otherwise, install the core runtime packages below:

```bash
pip install streamlit fastapi uvicorn requests pydantic python-dotenv sqlalchemy psycopg2-binary
# Optional (for nicer UI in the Hub/App)
pip install streamlit-extras
# For the client dashboard (if you maintain a separate list)
pip install -r src/dashboard-requirements.txt  # optional if present
```

3) Configure environment

Create a `.env` at the project root if needed:

```env
# Backend bind (optional; defaults are fine for local use)
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000

# Database URL (example for PostgreSQL; adjust as needed)
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/cx_db

# Any other keys your pipeline/components need
# OPENAI_API_KEY=...
```

You can also place sensitive values in Streamlit secrets if you prefer: `app/.streamlit/secrets.toml`.

---

## Run (Recommended): Streamlit Hub

Start the Hub; it will manage and link out to the apps.

```bash
streamlit run DashBoard_streamlit.py
```

From the Hub you can:
- Start/Restart the Customer App (review intake) and Client Dashboard (KPIs).
- See basic health/port status and open apps in new tabs.
- Optionally embed a live preview inside the Hub.

---

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
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
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

## Troubleshooting
- Port already in use: The apps try to find a free port automatically. If something collides, change `BACKEND_PORT` or the Streamlit UI ports, or stop other processes using those ports.
- Backend not reachable: Wait a few seconds for startup. Check `logs/backend.log`. Ensure your `.env` is set and that your Python environment has all required packages.
- Database connection errors: Verify `DATABASE_URL` is correct and the DB is reachable.
- Missing extras: UI “extras” are optional; install `streamlit-extras` if you want enhanced cards/effects.

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
