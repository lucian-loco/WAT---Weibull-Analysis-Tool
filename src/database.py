#!/usr/bin/python3
import oracledb
import os

_db_handle = None

def get_cursor():
    global _db_handle

    if _db_handle is None:
        conn_params = {
            'user':         os.environ['DB_USER'],
            'password':     os.environ['DB_PASS'],
            'host':         os.environ['DB_HOST'],
            'port':         os.environ['DB_PORT'],
            'service_name': os.environ['DB_SERV'],
        }

        _db_handle = oracledb.connect(**conn_params)

    return _db_handle.cursor()
