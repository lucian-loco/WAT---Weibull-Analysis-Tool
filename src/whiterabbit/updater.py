#!/usr/bin/env python3
import connectivity
import config
import threading
import time
import logging
logger = logging.getLogger(__name__)

def update_ptp():
    while True:
        try:
            logger.info("Updating connectivity data from PTP (via Icinga)")
            conn = connectivity.ConnectivityIcinga(config.WR_GRANDMASTER, False)
            conn.process()
            conn.save_to_json(config.WR_CONN_OUTPUT_FILE_PTP)
        except Exception as e:
            logger.error(f"Error in PTP updater: {e}")

        time.sleep(config.WR_CONN_REFRESH_RATE_PTP)


def update_layoutdb():
    while True:
        try:
            logger.info("Updating connectivity data from LayoutDB")
            conn = connectivity.ConnectivityDatabase.from_layoutdb(config.WR_GRANDMASTER, False)
            conn.process()
            conn.save_to_json(config.WR_CONN_OUTPUT_FILE_LAYOUTDB)
        except Exception as e:
            logger.error(f"Error in LayoutDB updater: {e}")

        time.sleep(config.WR_CONN_REFRESH_RATE_LAYOUTDB)


logger.setLevel(logging.INFO)
t1 = threading.Thread(target=update_ptp, daemon=True)
t2 = threading.Thread(target=update_layoutdb, daemon=True)
t1.start()
t2.start()
t1.join()
t2.join()
