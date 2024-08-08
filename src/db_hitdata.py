#!/usr/bin/python3
import oracledb
import os
import logging

logging.basicConfig(format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

_db_handle = None

def get_cursor():
    global _db_handle

    if _db_handle is None or not _db_handle.is_healthy():
        conn_params = {
            'user':         os.environ['DB_USER'],
            'password':     os.environ['DB_PASS'],
            'host':         os.environ['DB_HOST'],
            'port':         os.environ['DB_PORT'],
            'service_name': os.environ['DB_SERV'],
        }

        logger.info('Connecting to the database')
        _db_handle = oracledb.connect(**conn_params)

    return _db_handle.cursor()
