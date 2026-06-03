"""
Python script to update Confluence page with database query table + calculated values

This script:
1. Queries your database for reliability metrics
2. Calculates additional values in Python
3. Creates a Confluence table
4. Updates the Confluence page automatically
"""

import os
import io
import db_hitdata
import pandas as pd
from typing import Any
from textwrap import dedent
from datetime import datetime
from confluence_api import Confluence
from weibull import get_forecast_cache
from utils import get_logger

logger = get_logger(__name__)



def format_value_for_confluence(value: Any) -> str:
    """
    Format a value for display in Confluence table.

    - Converts None to empty string
    - Converts all values to strings (Python does this automatically in f-strings)

    Args:
        value: Any value from database or calculation

    Returns:
        String formatted for Confluence HTML table
    """
    if value is None:
        return ""

    if isinstance(value, float):
        if pd.isna(value):
            return ""

    # Convert to string
    value_str = str(value)

    # Simple version (no color-coding):
    return value_str


def format_forecast_cell(fr):
    if fr is None:
        return "Not available"

    lower = fr.get("lower_bound")
    expected = fr.get("expected_failures")
    upper = fr.get("upper_bound")

    if lower is None or expected is None or upper is None:
        return "Not available"

    return f"Upper: {upper:.1f}<br/>Expected: {expected:.1f}<br/>Lower: {lower:.1f}"


def make_html_table(f, rows):
    """
    Create HTML table exactly like your colleague's wrenscan.py

    Args:
        f: File-like object to write to (e.g., io.StringIO())
        rows: List of dictionaries with table data
    """
    if not rows:
        f.write("<p>No data available. Query to data base was not successful.</p>\\n")
        return

    columns = list(rows[0].keys())

    f.write("<table><thead>\\n<tr>")
    for col in columns:
        f.write(f"<th>{col}</th>")

    f.write("</tr>\\n</thead><tbody>\\n")
    for row in rows:
        f.write("<tr>")
        for col in columns:
            value = format_value_for_confluence(row.get(col))
            f.write(f"<td>{value or ''}</td>")
        f.write("</tr>\\n")
    f.write("</tbody></table>\\n")


def build_forecast_lookup(forecast_results):
    lookup = {}
    for part, forecast in forecast_results.items():
        rows = forecast.get("results", [])
        lookup[part] = {float(r["delta"]): r for r in rows}
    return lookup


def enrich_query_rows_with_forecast(df_query, forecast_lookup, deltas):
    out = df_query.copy()

    for d in deltas:
        out[f"EXPECTED FAILURES +{int(d)}D"] = "Not available"

    for idx, row in out.iterrows():
        part = row["EQUIPMENT CODE"]
        part_forecasts = forecast_lookup.get(part)

        if not part_forecasts:
            continue

        for d in deltas:
            fr = part_forecasts.get(float(d))
            if fr is None:
                continue

            out.at[idx, f"EXPECTED FAILURES +{int(d)}D"] = format_forecast_cell(fr)

    return out


def reliability_summary_table():
    """
    Main function to fetch data, calculate values, and update Confluence page.
    """
    confluence_token = os.environ.get("CONFLUENCE_TOKEN")
    if not confluence_token:
        raise RuntimeError('CONFLUENCE_TOKEN not found.')

    # === CONFIGURATION ===
    SPACE_KEY = "HWI"  # Confluence space key
    ANCESTOR_ID = "470357223"  # parent page ID
    PAGE_TITLE = "Summary of Reliability Data"  # page title
    FORECAST_DELTAS = [90.0, 180.0, 365.0, 1095.0, 1825.0]

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
