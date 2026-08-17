# WAT - Weibull Analysis Tool

## Introduction

This repository contains Python scripts for Weibull reliability analysis, served through a Flask web server. It fetches failure and suspension (censored) running-time data for equipment parts from an Oracle database, fits Weibull models, selects the best model per part, and exposes probability plots and failure forecasts through simple HTTP endpoints.

A `Dockerfile` is included to run the server inside a container, and the resulting container images are stored in the registry associated with this repository.

### Repository structure

The tool only uses the Python scripts whose file names start with a **lowercase** letter (e.g. `main.py`, `weibull.py`, `data_weibull.py`, ...). These are the modules actually imported and run by the Flask server.

Scripts whose file names start with an **uppercase** letter (e.g. `DeltaCV_Tuning.py`, `Predictive_Accuracy.py`, `Synthetic_Data.py`, `Validate_Weibull_CI.py`) are **not** part of the running tool. They were used for the analysis, validation, and tuning work behind my master's thesis (e.g. cross-validation delta tuning, predictive-accuracy studies, synthetic-data generation, confidence-interval validation) and are kept in the repository for completeness/reproducibility only. You can ignore them when running or deploying the tool.

## How it works

1. **Data access** (`db_hitdata.py`, `data_weibull.py`) — connects to the Oracle database (`weibull_data` view) and retrieves failure/suspension running times per part. Parts need a minimum number of failures and distinct failure times to qualify for analysis. Results are cached in memory to avoid hitting the database on every request.
2. **Model fitting and selection** (`weibull.py`, `weibull_evaluation.py`) — fits candidate Weibull models per part and selects the best one using cross-validation (CV, default), BIC (falback), or AICc.
3. **Confidence bounds** (`weibull_ci.py`) — computes confidence intervals for the fitted parameters/curves (analytical Fisher-Matrix method as default, additional available: bootstrap and parametric Monte Carlo method).
4. **Forecasting** (`weibull_forecast.py`) — projects the expected number of failures over configurable time horizons, with confidence bounds.
5. **Web server** (`main.py`) — a Flask app that ties everything together, serves plots and forecasts on demand, and refreshes all caches automatically once a day.
6. **Reporting** (`reliability_confluence_summary.py`, `confluence_api.py`) — optionally publishes a summary reliability table to a Confluence page.

Supporting modules: `utils.py` (logging, input validation, custom exceptions) and `weibull_user_input.py` (interactive CLI prompts for the underlying validation helpers, useful for local/manual analysis outside the web server).

## Endpoints

Below you will find the endpoints provided by the server. Once the server is up, you can access them by opening `http(s)://your-server-address/**endpoint**`. If an endpoint requires parameters, pass them as `http(s)://your-server-address/endpoint?**param1=value1&param2=value2**`, etc.

### `/`

Landing page of the tool. Shows basic build information (build date and git commit hash, read from environment variables).

### `/weibull`

Generates a Weibull probability plot for a given part.

**Method:** `GET`, `POST`

**Parameters:**

| Parameter | Required | Description |
|---|---|---|
| `part` | Yes | Equipment code for the part to be analyzed (e.g. `HCCTRI`). |
| `plot_type` | Yes | `CDF` for a failure-probability plot, or `SF` for a survival/reliability plot. |
| `edit` | No | Set to `1` to open an editable form (select model-selection method and confidence level) instead of generating a plot immediately with the defaults. |

**Form fields** (used when submitting via the `edit` form, `POST` only):

| Field | Description | Default |
|---|---|---|
| `sort_by` | Model-selection method: `CV` (cross-validation), `BIC`, or `AICc`. | `CV` |
| `ci` | Confidence level, in `[0, 1)`; `0` disables the confidence interval. | `0.95` |

**Returns:** an HTML page with an embedded PNG image showing the Weibull probability plot for the requested part. Returns HTTP 400 with a plain-text error if `part`/`plot_type` are missing or invalid, or if the plot cannot be generated (e.g. not enough data for that part).

**Example:**

```
http://localhost:8888/weibull?part=HCCTRI&plot_type=CDF
http://localhost:8888/weibull?part=HCCTRI&plot_type=SF&edit=1
```

### `/forecast`

Computes the expected number of failures (with confidence bounds) for a given part, over one or more future time horizons.

**Method:** `GET`, `POST`

**Parameters:**

| Parameter | Required | Description |
|---|---|---|
| `part` | Yes | Equipment code for the part to be analyzed. |
| `edit` | No | Set to `1` to open an editable form instead of computing the forecast immediately with the defaults. |

**Form fields** (used when submitting via the `edit` form, `POST` only):

| Field | Description | Default |
|---|---|---|
| `sort_by` | Model-selection method: `CV`, `BIC`, or `AICc`. | `CV` |
| `fc` | Comma-separated forecast horizons, in days. | `365, 730, 1095, 1825` |
| `ci` | Confidence level, in `[0, 1)`. | `0.95` |

**Returns:** an HTML page listing the expected number of failures (with lower/upper confidence bounds) for each requested horizon, together with the selection method actually used and today's date. Returns HTTP 400 with a plain-text error if `part` is missing/invalid or if the forecast cannot be computed.

**Example:**

```
http://localhost:8888/forecast?part=HCCTRI
http://localhost:8888/forecast?part=HCCTRI&edit=1
```

## Caching and scheduled refresh

To keep response times low, the server keeps three in-memory caches that are refreshed together:

1. Raw failure/suspension data (pulled from the database).
2. Model-selection results (best-fit model per part, using cross-validation by default).
3. Expected-failures forecasts per part.

These caches are populated once, synchronously, at application startup, and then refreshed automatically every day at 01:00 (Europe/Zurich server time) via a background scheduler (APScheduler). Caching can be disabled with the `WEIBULL_CACHE_ENABLED` environment variable (see below), in which case every request queries the database directly.

## Environment variables

This tool uses database queries to fetch data. To connect to the database, set the following environment variables:

| Variable | Description |
|---|---|
| `DB_USER` | Database user |
| `DB_PASS` | Database password |
| `DB_HOST` | Connection address |
| `DB_PORT` | Connection port |
| `DB_SERV` | Service name |

Optional environment variables:

| Variable | Description | Default |
|---|---|---|
| `WEIBULL_CACHE_ENABLED` | Set to `false`/`off`/`0`/`no` to disable in-memory caching and query the database directly on every request. Any other value (or unset) keeps caching enabled. | `true` |
| `CONFLUENCE_TOKEN` | Authentication token used by `reliability_confluence_summary.py` to publish the reliability summary table to Confluence. Required only if that reporting step is enabled. | — |
| `DRAWIO_EXPORT_URL` | URL of an optional draw.io export server, used by the app. | `''` |
| `APP_BUILD_DATE` | Build date shown on the landing page. Typically set at image build time. | `unknown` |
| `APP_GIT_COMMIT` | Git commit hash shown on the landing page. Typically set at image build time. | `unknown` |

When the project runs on an OpenShift instance, use the `Secrets` or `ConfigMaps` mechanisms to set these environment variables rather than hard-coding them.

## Running locally (Python)

1. Create a new virtual environment: `python3 -m venv venv`
2. Activate the environment: `source venv/bin/activate`
3. Install dependencies: `pip3 install -r requirements.txt`
4. Set the required environment variables (`DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`, `DB_SERV`, and optionally `WEIBULL_CACHE_ENABLED`, `CONFLUENCE_TOKEN`).
5. Modify the code as needed.
6. Start the server: `python3 src/main.py`
7. Navigate to `http://localhost:8888/**endpoint**` to test the changes (e.g. `http://localhost:8888/weibull?part=HCCTRI&plot_type=CDF`).

The server listens on port `8888` by default and runs with `debug=True`. Note that when caching is enabled, the Flask auto-reloader is disabled on purpose, so the app doesn't reload (and re-fetch the whole cache) on every code change — restart the process manually to pick up changes.

## Running with Docker

1. Modify the `Dockerfile` if needed.
2. Rebuild and run the image: `docker_start.sh`

The resulting container image is published to the container registry associated with this repository and can be deployed as-is (e.g. on OpenShift), provided the required environment variables are supplied via `Secrets` or `ConfigMaps`.

## Notes on the thesis-only scripts

The uppercase-named scripts (`DeltaCV_Tuning.py`, `Predictive_Accuracy.py`, `Synthetic_Data.py`, `Validate_Weibull_CI.py`) are standalone analysis scripts developed as part of the underlying master's thesis with the title "Development of a Universally Applicable Tool for Robust Weibull Analysis of Field Systems -- Applied to CERN's Front-End Computers" by Lucian Groha. They are not imported by `main.py` and are not required to run the web tool. They are kept here for transparency and reproducibility of the statistical methodology (e.g. tuning the CV delta parameter, evaluating predictive accuracy, generating synthetic datasets for validation, and validating confidence-interval coverage), and can be run independently if you want to reproduce or extend that analysis.
