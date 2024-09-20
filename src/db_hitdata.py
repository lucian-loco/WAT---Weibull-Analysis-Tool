#!/usr/bin/python3
import oracledb
import os
import logging

logging.basicConfig(format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

__db_conn_params = {
    'user':         os.environ['DB_USER'],
    'password':     os.environ['DB_PASS'],
    'host':         os.environ['DB_HOST'],
    'port':         os.environ['DB_PORT'],
    'service_name': os.environ['DB_SERV'],
}

def get_cursor():
    db_handle = oracledb.connect(**__db_conn_params)
    return db_handle.cursor()
