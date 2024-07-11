#!/usr/bin/env python3
import connectivity
import threading
import time
import logging
logger = logging.getLogger(__name__)

WR_GRANDMASTER = "ctdwa-ccr-cgpnallm1"  # TODO change to the actual WR Grandmaster

REFRESH_RATE_PTP = 5 * 60
OUTPUT_FILE_PTP = "./data/connectivity_ptp.json"
REFRESH_RATE_LAYOUTDB = 2 * 60 * 60
OUTPUT_FILE_LAYOUTDB = "./data/connectivity_layoutdb.json"

def update_ptp():
    while True:
        try:
            logger.info("Updating connectivity data from PTP")
            conn = connectivity.ConnectivityPTP(WR_GRANDMASTER, False)
            conn.process()
            conn.save_to_json(OUTPUT_FILE_PTP)
        except Exception as e:
            logger.error(f"Error in PTP updater: {e}")

        time.sleep(REFRESH_RATE_PTP)


def update_layoutdb():
    while True:
        try:
            logger.info("Updating connectivity data from LayoutDB")
            conn = connectivity.ConnectivityDatabase.from_layoutdb(WR_GRANDMASTER, False)
            conn.process()
            conn.save_to_json(OUTPUT_FILE_LAYOUTDB)
        except Exception as e:
            logger.error(f"Error in LayoutDB updater: {e}")
    
        time.sleep(REFRESH_RATE_LAYOUTDB)


logger.setLevel(logging.INFO)
t1 = threading.Thread(target=update_ptp, daemon=True)
t2 = threading.Thread(target=update_layoutdb, daemon=True)
t1.start()
t2.start()
t1.join()
t2.join()