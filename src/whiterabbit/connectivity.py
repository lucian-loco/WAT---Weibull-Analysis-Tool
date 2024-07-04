#!/usr/bin/env python3
import sys
import os.path
# add path to import modules from the directory above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
import logging
import argparse
import db_hitdata
import ccda
import csv

from switch import Switch
from devices import *

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

WR_GRANDMASTER = "ctdw-ccr-ctnlmj1"

class Connectivity:
    def __init__(self, top_switch, all_connections=False):
        self._grandmaster = top_switch
        self._all_connections = all_connections

        self._devices = {}
        self._ports = {}
        self._port_connections = {}
        self._fibers = []                   # TODO try to get rid of it

        # visjs data
        self.edges = []
        self.nodes = []
    

    def get_device(self, name):
        name = name.lower()

        if name not in self._devices:
            self._devices[name] = Device(name)

        return self._devices[name]


    def get_port(self, name):
        if name not in self._ports:
            self._ports[name] = Port(name)

        return self._ports[name]


    def device_description(self, device):
        ret = "Master: {0}\n".format(device.master)
        #ret += "Layer: {0}\n".format(device.layer)

        for p in device.sorted_ports():
            try:
                other_end = self._port_connections[p].device
            except:
                other_end = None

            ret += "Port {0} -> {1} ({2})\n".format(p.label, other_end, p.name)

        return ret


    def fiber_description(self, fiber):
        return str(fiber)


    # Returns port connected to this_port on the other connection end
    # E.g. if port A is connected to port B, then get_peer_port(A) == B
    def get_peer_port(self, this_port):
        return self._ports[self._port_connections[this_port].name]


    # Returns device connected to this_port on the other connection end
    def get_peer_device(self, this_port):
        return self._devices[self.get_peer_port(this_port).device]


    def _add_fiber(self, fiber):
        self._fibers.append(fiber)
        self._port_connections[fiber.start] = fiber.end
        self._port_connections[fiber.end] = fiber.start


    def _assign_layers_full(self, top, layer=1):
        assert(top.layer == 0 and top.id is None)
        next_queue = [top]

        while next_queue:
            queue = next_queue
            next_queue = []

            while queue:
                device = queue.pop(0)

                if not device.is_switch():
                    continue

                #if device.id is not None:  # uncommenting helps spread devices between layers
                #    continue

                logger.debug(">> PROCESSING {0} LAYER={1}".format(device.name, layer))

                device.id = None
                device.assign_id()
                device.layer = layer

                for port in device.ports.values(): #sorted_ports():
                    try:
                        # See what is connected on the other end
                        other_device = self.get_peer_device(port)

                        if not other_device.is_switch():
                            continue

                        if other_device.id is None and other_device not in next_queue:    # this check prevents looping back to the master
                            logger.debug("  CHILD {0}".format(other_device.name))
                            next_queue.append(other_device)

                        # Master switch is normally connected to port number one
                        if port.label == 1:
                            assert(device.master == None or device.master == other_device.name)
                            device.master = other_device.name

                    except KeyError as e:
                        logger.debug('No switch connected to {0}'.format(port))

            layer += 1


    def _assign_layers_hierarchy(self, top, layer=1):
        assert(top.layer == 0 and top.id is None)
        next_queue = [top]
        top.layer = layer

        while next_queue:
            queue = next_queue
            next_queue = []

            while queue:
                device = queue.pop(0)

                if device.id:
                    continue    # device already processed

                device.assign_id()

                for port in device.ports.values(): #sorted_ports():
                    try:
                        # See what is connected on the other end
                        other_device = self.get_peer_device(port)

                        if not other_device.is_switch():
                            continue

                        # Master switch is normally connected to port number one
                        if port.label == 1:
                            assert(device.master == None or device.master == other_device.name)
                            device.master = other_device.name
                            device.layer = other_device.layer + 1
                        elif other_device.id is None:
                            next_queue.append(other_device)

                    except KeyError as e:
                        logger.debug('No switch connected to {0}'.format(port))


    def process(self):
        assert(len(self.edges) == 0 and len(self.nodes) == 0)

        grandmaster = self._devices[self._grandmaster]

        if self._all_connections:
            self._assign_layers_full(grandmaster)
        else:
            self._assign_layers_hierarchy(grandmaster)

        # Prepare data for visjs graph
        # Nodes
        for device in self._devices.values():
            # If there is no ID assigned, then this device is outside the main WR network
            if device.id is None:
                continue

            # There are some fibers which have "None" as the parent label
            # in such case they seem connected to the same device ("None"),
            # which is obviously not true
            if device.name == "None":
                continue

            self.nodes.append({
                "id": device.id,
                "label": device.name,
                "level": device.layer,
                "title": self.device_description(device),
            })


        # Edges
        if self._all_connections:
            for fiber in self._fibers:
                try:
                    device1_name = self._ports[fiber.start.name].device
                    device2_name = self._ports[fiber.end.name].device

                    if device1_name is None or device2_name is None:
                        continue

                    device1 = self._devices[device1_name]
                    device2 = self._devices[device2_name]
                    self.edges.append({
                        "from": device1.id, "to": device2.id,
                        "title": self.fiber_description(fiber)
                    })
                except:
                    logger.info("Could not generate edge: {0}".format(fiber))
        else:
            for device in self._devices.values():
                if device.master:
                    master = self._devices[device.master]
                    self.edges.append({ "from": device.id, "to": master.id })   # TODO description?


class ConnectivityDatabase(Connectivity):
    """ Class generating connectivity information from a database/CSV file. """
    def __init__(self, top_switch, all_connections=False):
        super(ConnectivityDatabase, self).__init__(top_switch, all_connections)
        self._incomplete_fibers = []


    @staticmethod
    def from_csv(filename, *args, **kwargs):
        instance = ConnectivityDatabase(*args, **kwargs)

        with open(filename) as f:
            reader = csv.DictReader(f)

            for row in reader:
                instance._process_row(row)

        return instance


    @staticmethod
    def from_layoutdb(*args, **kwargs):
        # Note that there are more column available in the view,
        # but they are not used by the script
        columns = ("PRIME_NAME", "PRIME_LABEL", "PRIME_PARENT_NAME",
                "PRIME_PARENT_LABEL", "SECOND_NAME", "SECOND_LABEL",
                "SECOND_PARENT_NAME", "SECOND_PARENT_LABEL")

        cursor = db_hitdata.get_cursor()
        sql_query = "SELECT {0} FROM controls_wr_fibres_v".format(", ".join(columns))
        result = cursor.execute(sql_query)

        instance = ConnectivityDatabase(*args, **kwargs)

        for row in result:
            # Build a dictionary with columns as the keys
            row_annotated = {columns[i]: row[i] for i in range(len(columns))}
            instance._process_row(row_annotated)

        return instance


    def _process_row(self, row):
        if row["PRIME_NAME"] == row["SECOND_NAME"]:
            logger.warning("Loop detected: {0}".format(row["PRIME_NAME"]))
            return


        # Process ports
        for pfx in ("PRIME", "SECOND"):
            device_name = row["{0}_PARENT_LABEL".format(pfx)]
            port_name = row["{0}_NAME".format(pfx)]
            port_label = row["{0}_LABEL".format(pfx)]

            port = self.get_port(port_name)
            port.label = port_label
            port.device = device_name

            if port.is_wrs_port:
                try:
                    device = self.get_device(device_name)

                    if device.is_switch():
                        # Normally label indicates the port number, but not always
                        # (most nodes have a text description, so cannot be casted to int)
                        # this cast helps with proper port sorting
                        port.label = int(port_label)

                    device.add_port(port)
                except:
                    logger.info("Skipping port '{0}' for device '{1}'".format(port_label, device_name))


        # Process fiber connection
        port1 = self.get_port(row["PRIME_NAME"])
        port2 = self.get_port(row["SECOND_NAME"])
        new_fiber = Fiber(port1, port2)

        if new_fiber.is_complete:
            # This fiber connects two switches directly, no more processing needed
            self._add_fiber(new_fiber)
        else:
            # It is a fiber between a switch and a patch panel or between two patch panels,
            # so try to merge it with another fiber in order to make a connection between two switches
            self._incomplete_fibers.append(new_fiber)
            self._try_to_merge_fiber(new_fiber)


    def _try_to_merge_fiber(self, fiber):
        for f in self._incomplete_fibers:
            if f == fiber:
                continue    # do not try to merge a fiber to itself

            if f.can_merge(fiber):
                self._incomplete_fibers.remove(fiber)
                self._incomplete_fibers.remove(f)
                f.merge(fiber)

                if f.is_complete:
                    self._add_fiber(f)
                else:
                    self._incomplete_fibers.append(f)

                return True

        return False


    def _process_incomplete_fibers(self):
        merged_sth = True

        while merged_sth:
            merged_sth = False

            for f in self._incomplete_fibers:
                if self._try_to_merge_fiber(f):
                    merged_sth = True
                    # Restart the loop, _incomplete_fibers list has been modified
                    break


    def process(self):
        self._process_incomplete_fibers()
        super(ConnectivityDatabase, self).process()


class ConnectivityPTP(Connectivity):
    """ Class generating connectivity information from SNMP/PTP (not LLDP). """
    def __init__(self, top_switch, all_connections=False):
        super(ConnectivityPTP, self).__init__(top_switch, all_connections)
        self._sfp_port_count = 18                   # TODO do not hardcode, anticipate 24 port switches
        self._build_mac_db()


    def get_device(self, name):
        # Nearly the same as Connectivity.get_device() but this one creates switches
        name = name.lower()

        if name not in self._devices:
            self._devices[name] = Switch(name)

        return self._devices[name]


    def get_port_by_mac(self, mac):
        try:
            return self._ports[ConnectivityPTP._sanitize_mac(mac)]
        except KeyError:
            return None


    def get_switch_by_mac(self, mac):
        port = self.get_port_by_mac(mac)

        if port is None or not port.device:
            return None

        try:
            return self._devices[port.device]
        except KeyError:
            return None


    def _build_mac_db(self):
        for entry in ccda.wr_switches():
            switch_name = entry["name"]

            try:
                switch = self.get_device(switch_name)
            except:
                logger.warning(f"Could not connect to {switch_name}")
                continue

            try:
                # Get the MAC address of the first SFP port (remaining SFP ports have consecutive addresses)
                mac1 = switch.sfp_port_mac(1)
                # Convert the MAC address to an int, to easily compute other port MAC addresses
                mac1_num = ConnectivityPTP._mac_to_int(mac1)

                # Bind all MACs to the switch
                for p in self._sfp_port_range():
                    # Compute the port MAC address, store in hex format
                    mac_str = ConnectivityPTP._compute_mac(mac1_num, p)

                    # Register the port, to create connections
                    assert(mac_str not in self._ports)  # Check for duplicates
                    new_port = self.get_port(mac_str)
                    new_port.label = p             # Port number, indexed from 1
                    new_port.device = switch_name
                    switch.add_port(new_port)
                    logger.debug(f"MAC DB: {mac_str} -> {switch_name}")
            except ConnectionError:
                logger.warning(f"Could not connect to {switch_name}")


    def _process_switch(self, switch: Switch):
        logger.debug(f"Processing {switch.name}")

        # Get the first SFP port MAC address as an integer (to compute MAC addresses of the remaining ports)
        my_mac1_num = ConnectivityPTP._mac_to_int(switch.sfp_port_mac(1))

        # Check what is connected to each SFP port...
        for p in self._sfp_port_range():
            peer_mac = switch.sfp_port_peer_mac(p)

            if peer_mac == None:
                continue        # nothing connected to the port

            try:
                my_mac = ConnectivityPTP._compute_mac(my_mac1_num, p)
                my_port = self._ports[my_mac]
                peer_port = self._ports[peer_mac]
                self._add_fiber(Fiber(my_port, peer_port))
                logger.debug(f"   {my_port} <-> {peer_port}")
            except KeyError:
                logger.warning(f"Unknown MAC address {peer_mac} on {my_port}")


    def process(self):
        # Add fibers basing on the LLDP data
        for s in self._devices.values():
            try:
                self._process_switch(s)
            except ConnectionError:
                logger.warning(f"Could not process {s.name}")

        super(ConnectivityPTP, self).process()


    @staticmethod
    def _sanitize_mac(mac):
        """ Converts MAC address to the format used by the class. """
        sanitized = mac.replace(':', '').replace('-', '').lower()
        assert(len(sanitized) == 12)
        return sanitized


    @staticmethod
    def _int_to_mac(mac_int):
        """ Converts an integer to a hexadecimal number representing a MAC address. """
        return '{0:012x}'.format(mac_int)


    @staticmethod
    def _mac_to_int(mac_str):
        """ Converts a string representing a (sanitized) MAC address to an integer. """
        return int(mac_str, 16)


    @staticmethod
    def _compute_mac(mac1_num, port_index):
        """ Calculates MAC address using first SFP port MAC address and the requested SFP port number (indexed from 1). """
        return ConnectivityPTP._int_to_mac(mac1_num + port_index - 1)


    def _sfp_port_range(self):
        """ Returns the range used to iterate through SFP ports. """
        return range(1, self._sfp_port_count + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates WR connectivity information (JSON format)")
    
    parser.add_argument("--source", "-s", choices=("layoutdb", "ptp", "csv"), required=True,
            help="Data source.")
    parser.add_argument("--output", "-o", type=str, default="connectivity.json",
            help="Output file name.")
    parser.add_argument("--verbose", "-v", action="store_true",
            help="Enable verbose output")
    parser.add_argument("--all", action="store_true",
            help="Include all connections (may create a non-tree structure).")
    parser.add_argument("--top-switch", default=WR_GRANDMASTER,
            help="Switch which should be used as the top of the tree (grandmaster).")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.source == "layoutdb":
        conn = ConnectivityDatabase.from_layoutdb(args.top_switch, all_connections=args.all)
    elif args.source == "csv":
        conn = ConnectivityDatabase.from_csv(args.top_switch, "connections.csv", all_connections=args.all)
    elif args.source == "ptp":
        conn = ConnectivityPTP(args.top_switch, all_connections=args.all)

    conn.process()
    # logger.info("Incomplete fibers: {0}".format(len(conn._incomplete_fibers)))
    logger.info("Fibers: {0} Devices: {1}".format(len(conn._fibers), len(conn._devices)))

    #for name, device in conn._devices.items():
    #    print("> {0}".format(name))
    #    for port_number, port_data in device._ports.items():
    #        print(port_data)

    # Save graph data
    with open(args.output, "w") as output:
        json.dump({"edges": conn.edges, "nodes": conn.nodes}, output)