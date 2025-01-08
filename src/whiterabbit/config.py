#!/usr/bin/python3
import os
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

WR_GRANDMASTER = os.environ.get("WR_GRANDMASTER", "ctdw-ccr-cgm1")

WR_CONN_REFRESH_RATE_PTP = int(os.environ.get("WR_CONN_REFRESH_RATE_PTP", 5 * 60))
WR_CONN_OUTPUT_FILE_PTP = os.environ.get("WR_CONN_OUTPUT_FILE_PTP", os.path.join(_SCRIPT_DIR, "data/connectivity_ptp.json"))

WR_CONN_REFRESH_RATE_LAYOUTDB = int(os.environ.get("WR_CONN_REFRESH_RATE_LAYOUTDB", 2 * 60 * 60))
WR_CONN_OUTPUT_FILE_LAYOUTDB = os.environ.get("WR_CONN_OUTPUT_FILE_LAYOUTDB", os.path.join(_SCRIPT_DIR, "data/connectivity_layoutdb.json"))
