#!/usr/bin/python3
import oracledb
import os
import logging

logging.basicConfig(format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

try:
    __pool_size = os.environ['GUNICORN_THREADS'] + 1
except KeyError:
    __pool_size = 2

__pool = oracledb.create_pool(
    user=os.environ['DB_USER'],
    password=os.environ['DB_PASS'],
    host=os.environ['DB_HOST'],
    port=os.environ['DB_PORT'],
    service_name=os.environ['DB_SERV'],
    min=__pool_size,
    max=__pool_size
)

logger.info('Created HIT database connection pool (size=%d)', __pool_size)


def get_cursor():
    global __pool
    conn = __pool.acquire()
    return conn.cursor()
