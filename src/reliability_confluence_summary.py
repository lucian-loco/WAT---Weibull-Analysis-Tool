#!/usr/bin/python3
"""
reliability_confluence_summary.py
==================================

Python script to update Confluence page with database query table + calculated values.

This script:
1. Queries the reliability database (`reliability_part_metrics` joined with the hardware catalogue)
   for per-part asset counts, failure counts, installation-time statistics, and MTBF.
2. Enriches each row with expected-failure forecast values (point estimate, lower/upper confidence bounds)
   for configured time horizons, sourced from the Weibull forecast cache (`weibull.get_forecast_cache`).
3. Renders the combined result as an HTML table wrapped in a Confluence "html" macro (with inline CSS styling and an
   auto-generation notice).
4. Publishes (creates or updates) the resulting page on Confluence via the `confluence_api.Confluence` client.

Intended to be run as a scheduled/CLI job after the Weibull data, analysis, and forecast caches have been refreshed
(see the `__main__` block), so that `get_forecast_cache()` returns up-to-date results.

Configuration (hard-coded within `reliability_summary_table`):
    SPACE_KEY     : Confluence space key ("HWI").
    ANCESTOR_ID   : Parent page ID under which the report page is nested.
    PAGE_TITLE    : Title of the Confluence page to create/update.
    FORECAST_DELTAS : Forecast horizons (in days) to display as extra columns (currently 1 year and 5 years).
    SQL_QUERY     : The reliability metrics query executed against the database.

Environment variables:
    CONFLUENCE_TOKEN : str, required
        Authentication token for the Confluence API client. The script raises a RuntimeError if this is not set.

Author: Lucian Groha
"""
import os
import io
import html
import db_hitdata
import pandas as pd
from typing import Any
from textwrap import dedent
from datetime import datetime
from urllib.parse import quote
from confluence_api import Confluence
from weibull import get_forecast_cache
from utils import get_logger

logger = get_logger(__name__)

DASHBOARD_URL_TEMPLATE = ("https://confluence.cern.ch/spaces/HWI/pages/753809310/HIT+Reliability+Dashboard?pageId=753809310&run_1_part_code={code}&run_1=run")



def format_value_for_confluence(value: Any) -> str:
    """
    Format a single value for safe display in an HTML table cell on Confluence.

    Parameters
    ----------
    value : Any
        A value from the database query result or a calculated forecast field (e.g. int, float, str, None, or NaN).

    Returns
    -------
    str
        "-" if the value is None or NaN; a one-decimal-place string for floats;
        a plain string for ints; and the string representation for anything else. All string output is HTML-escaped via
        `html.escape` to prevent malformed or unsafe markup in the Confluence page.
    """
    if value is None or pd.isna(value):
        return "-"

    if isinstance(value, float):
        return html.escape(f"{value:.1f}", quote=True)

    if isinstance(value, int):
        return html.escape(str(value), quote=True)

    return html.escape(str(value), quote=True)


def make_equipment_code_link(part_code: str) -> str:
    url = DASHBOARD_URL_TEMPLATE.format(code=quote(str(part_code)))
    label = html.escape(str(part_code), quote=True)

    return f'<a href="{html.escape(url, quote=True)}">{label}</a>'


def make_html_table(f, rows):
    """
    Write an HTML table (header + body rows) representing tabular data to a file-like object,
    for embedding in a Confluence page via the HTML macro.

    Parameters
    ----------
    f : file-like object
        Writable stream (e.g. `io.StringIO()`) that the HTML markup is written to.
    rows : list[dict]
        List of row dictionaries; the keys of the first row determine the table's column order and headers.
        Each cell value is formatted via `format_value_for_confluence`.

    Returns
    -------
    None
        Writes directly to `f`. If `rows` is empty, writes a "No data available" placeholder paragraph
        instead of a table.
    """
    if not rows:
        f.write("<p>No data available. Query to data base was not successful.</p>\n")
        return

    link_column = 'EQUIPMENT CODE'

    columns = list(rows[0].keys())

    f.write("<table><thead>\n<tr>")
    for col in columns:
        f.write(f"<th>{format_value_for_confluence(col)}</th>")
    f.write("</tr>\n</thead><tbody>\n")

    for row in rows:
        f.write("<tr>")
        for col in columns:
            raw_value = row.get(col)
            if col == link_column and raw_value is not None and not pd.isna(raw_value):
                cell = make_equipment_code_link(raw_value)
            else:
                cell = format_value_for_confluence(raw_value)
            f.write(f"<td>{cell}</td>")
        f.write("</tr>\n")

    f.write("</tbody></table>\n")


def build_forecast_lookup(forecast_results):
    """
    Restructure the raw per-part forecast results into a nested lookup keyed by part name and forecast horizon (delta),
    for fast row-by-row enrichment of the query DataFrame.

    Parameters
    ----------
    forecast_results : dict
        Mapping of part name to a forecast dict (as produced by `weibull_forecast.forecast_part_direct_delta`),
        each containing a 'results' list of per-delta forecast entries (dicts with a 'delta' key among others).

    Returns
    -------
    dict[str, dict[float, dict]]
        Mapping: part name -> {delta (float): forecast entry dict}, where each forecast entry dict contains keys
        like 'expected_failures', 'lower_bound', 'upper_bound' (see `weibull_forecast._expected_failures_direct_delta`).
    """
    lookup = {}
    for part, forecast in forecast_results.items():
        rows = forecast.get("results", [])
        lookup[part] = {float(r["delta"]): r for r in rows}
    return lookup


def delta_label(d):
    years = int(d / 365)
    return f"{years} year" if years == 1 else f"{years} years"


def enrich_query_rows_with_forecast(df_query, forecast_lookup, deltas):
    """
    Add expected-failure forecast columns (upper bound, point estimate, lower bound) to the reliability metrics
    DataFrame, for each requested forecast horizon, matched by equipment/part code.

    Parameters
    ----------
    df_query : pandas.DataFrame
        Reliability metrics query result; must contain an "EQUIPMENT CODE" column used to look up forecasts per part.
    forecast_lookup : dict[str, dict[float, dict]]
        Nested forecast lookup as produced by `build_forecast_lookup`.
    deltas : list[float]
        Forecast horizons (in days) to add columns for.

    Returns
    -------
    pandas.DataFrame
        A copy of `df_query` with three new columns added per delta: "EXPECTED FAILURES +{label} UPPER", "... EXPECTED",
        and "... LOWER" (label from `delta_label`). Rows for parts with no matching forecast, or deltas with no matching
        forecast entry, are left as None in those columns.
    """
    out = df_query.copy()

    # Create 3 columns per delta
    for d in deltas:
        label = delta_label(d)
        out[f"EXPECTED FAILURES +{label} UPPER"] = None
        out[f"EXPECTED FAILURES +{label} EXPECTED"] = None
        out[f"EXPECTED FAILURES +{label} LOWER"] = None

    for idx, row in out.iterrows():
        part = row["EQUIPMENT CODE"]
        part_forecasts = forecast_lookup.get(part)

        if not part_forecasts:
            continue

        for d in deltas:
            fr = part_forecasts.get(float(d))
            if fr is None:
                continue

            label = delta_label(d)

            col = f"EXPECTED FAILURES +{label} UPPER"
            out.at[idx, col] = fr.get("upper_bound")

            col = f"EXPECTED FAILURES +{label} EXPECTED"
            out.at[idx, col] = fr.get("expected_failures")

            col = f"EXPECTED FAILURES +{label} LOWER"
            out.at[idx, col] = fr.get("lower_bound")

    return out


def reliability_summary_table():
    """
    Main entry point: fetch reliability metrics from the database, enrich them with Weibull expected-failure forecasts,
    render the combined result as an HTML table, and publish it to Confluence.

    Workflow
    --------
    1. Reads the `CONFLUENCE_TOKEN` environment variable (raises if missing).
    2. Executes the hard-coded `SQL_QUERY` against the reliability database via `db_hitdata.get_cursor()`.
    3. If the query returns rows, retrieves the pre-computed forecast cache (`get_forecast_cache()`),
       builds a per-part/per-delta lookup, and joins forecast columns (upper/expected/lower per configured horizon
       in `FORECAST_DELTAS`) onto the query result via `enrich_query_rows_with_forecast`.
    4. Reorders columns into a fixed display order: identification and count columns first, forecast columns
       in the middle, then age/installation-time/MTBF columns last.
    5. Renders the final table as HTML (via `make_html_table`), wrapped in a Confluence HTML macro with inline CSS
       styling and an auto-generation notice with the current timestamp.
    6. Publishes the page via `Confluence.insert_or_update_page`, creating or updating the page identified
       by `SPACE_KEY`, `ANCESTOR_ID`, and `PAGE_TITLE`.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The Confluence page is updated as a side effect; nothing is returned.

    Raises
    ------
    RuntimeError
        If the `CONFLUENCE_TOKEN` environment variable is not set.

    Notes
    -----
    If the SQL query returns no rows, the page is still published with an empty-data placeholder table,
    and a warning is logged.
    """
    confluence_token = os.environ.get("CONFLUENCE_TOKEN")
    if not confluence_token:
        raise RuntimeError('CONFLUENCE_TOKEN not found.')

    forecast_cols = []

    # === CONFIGURATION ===
    SPACE_KEY = "HWI"  # Confluence space key
    ANCESTOR_ID = "470357223"  # parent page ID
    PAGE_TITLE = "Summary of Reliability Data"  # page title
    FORECAST_DELTAS = [365.0, 1825.0]

    # SQL query
    SQL_QUERY = """select OBJ_PART "EQUIPMENT CODE", OBJ_DESC "DESCRIPTION", 
                          status, category, platform, function, 
                          NUM_ASSETS "NUMBER OF ASSETS", NUM_FAILURES "TOTAL NUMBER OF FAILURES", NUM_FIRST_FAILURES "NUMBER OF FIRST FAILURES",
                          NUM_UPGRADED_CENSORED "NUMBER OF UPGRADED ASSETS", NUM_INSTALLED_WITHOUT_FAILURE "NUMBER OF INSTALLED ASSETS WITHOUT FAILURE",
                          NUM_INSTALLED_WITH_FAILURE "NUMBER OF INSTALLED ASSETS WITH FAILURES IN THE PAST",
                          AVG_AGE "AVG AGE", TOTAL_INSTALLATION_TIME_YEARS "TOTAL INSTALLATION TIME IN YEARS",
                          AVG_INST_TIME_YEARS "AVG INSTALLATION TIME IN YEARS", to_number(decode(MTBF_YEARS, 'N/A', null, mtbf_years)) "MTBF IN YEARS"
                   from reliability_part_metrics a
                            JOIN hw_catalogue_v b on (a.obj_part = b.eam_part_code)
                   ORDER BY OBJ_PART"""

    # === FETCH DATA FROM DATABASE ===
    with db_hitdata.get_cursor() as cursor:
        cursor.execute(SQL_QUERY)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

    df_query = pd.DataFrame(rows, columns=columns)
    if df_query.empty:
        table_rows = []
        logger.warning("Query returned no rows.")
    else:
        forecast_pack = get_forecast_cache()
        forecast_lookup = build_forecast_lookup(forecast_pack['results'])

        df_final = enrich_query_rows_with_forecast(df_query, forecast_lookup, FORECAST_DELTAS)

        forecast_cols = []
        for d in FORECAST_DELTAS:
            label = delta_label(d)
            forecast_cols.extend([f"EXPECTED FAILURES +{label} UPPER",
                                  f"EXPECTED FAILURES +{label} EXPECTED",
                                  f"EXPECTED FAILURES +{label} LOWER"
            ])

        before_forecast = ["EQUIPMENT CODE",
                           "DESCRIPTION",
                           "STATUS",
                           "CATEGORY",
                           "PLATFORM",
                           "FUNCTION",
                           "NUMBER OF ASSETS",
                           "TOTAL NUMBER OF FAILURES",
                           "NUMBER OF FIRST FAILURES",
                           "NUMBER OF UPGRADED ASSETS",
                           "NUMBER OF INSTALLED ASSETS WITHOUT FAILURE",
                           "NUMBER OF INSTALLED ASSETS WITH FAILURES IN THE PAST"
        ]

        after_forecast = ["AVG AGE",
                          "TOTAL INSTALLATION TIME IN YEARS",
                          "AVG INSTALLATION TIME IN YEARS",
                          "MTBF IN YEARS"
        ]

        df_final = df_final[before_forecast + forecast_cols + after_forecast]

        table_rows = df_final.to_dict(orient="records")

    f = io.StringIO()
    # Write the HTML macro wrapper
    # fmt: off
    f.write(dedent(f"""\
            <ac:structured-macro ac:name="html" ac:schema-version="1" ac:macro-id="00b9303b-3d26-45f9-b597-9c59dba17d70">
              <ac:plain-text-body><![CDATA[
            <style>
            .confluenceTable .confluenceTd {{
                font-size: 12px;
                padding: 3px 6px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #dce8f0;
                padding: 8px 12px;
                text-align: left;
                vertical-align: top;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
            }}
            </style>
              ]]></ac:plain-text-body>
            </ac:structured-macro>
            <p><small><ac:emoticon ac:name="warning" />&ensp;This page is auto-generated from database query and Python forecasts.</small></p>
            <p><strong>Last updated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            """))
    # fmt: on

    # Add the table
    make_html_table(f, table_rows)

    c = Confluence(confluence_token)
    c.insert_or_update_page(space_key=SPACE_KEY, ancestor_id=ANCESTOR_ID, title=PAGE_TITLE, content=f.getvalue())



if __name__ == "__main__":
    from weibull import refresh_analysis_cache, refresh_forecast_cache
    from data_weibull import refresh_cache

    refresh_cache()  # 1. Pull from DB
    refresh_analysis_cache()  # 2. Model selection with CV (default)
    refresh_forecast_cache()  # 3. Expected failure forecasts

    try:
        reliability_summary_table()
    except Exception as e:
        print(f'ERROR: {e}')
