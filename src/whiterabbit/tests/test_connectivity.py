#!/usr/bin/env python3
import sys
sys.path.append("..")
from connectivity import Connectivity

# TODO turn it into a real pytest

def test_incomplete_fibers(conn):
    incomplete_nodes = set()

    # Verify that remaining incomplete fibers really cannot connect anything else
    for f in conn._incomplete_fibers:
        assert f.start not in incomplete_nodes
        assert f.end not in incomplete_nodes
        incomplete_nodes.add(f.start)
        incomplete_nodes.add(f.end)


conn = Connectivity.from_csv("connections.csv", all_connections=True)
test_incomplete_fibers(conn)
