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

import config
from switch import Switch
from devices import *

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class Connectivity:
    def __init__(self, top_switch, all_connections=False):
        self._grandmaster = top_switch
        self._all_connections = all_connections
        self.clear()


    def clear(self):
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


    def get_port(self, name, create=False):
        if name not in self._ports:
            if not create:
                return None

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


    def get_peer_port(self, this_port):
        """
          Returns port connected to this_port on the other connection end.
          E.g. if port A is connected to port B, then get_peer_port(A) == B.
        """
        return self._ports[self._port_connections[this_port].name]


    def get_peer_device(self, this_port):
        """ Returns device connected to this_port on the other connection end. """
        return self._devices[self.get_peer_port(this_port).device]


    def save_to_json(self, filename):
        """ Saves connectivity data to a JSON file. """
        with open(filename, "w") as output:
            json.dump({"edges": self.edges, "nodes": self.nodes}, output)


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
    SWITCH_PORT_PREFIX = "CTDNT"

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
        """
        Retrieves connectivity data from the Layout database.
        """
        # Note that there are more columns available in the view,
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

            port = self.get_port(port_name, True)
            port.label = port_label
            port.device = device_name

            if ConnectivityDatabase._is_wrs_port(port):
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
        port1 = self.get_port(row["PRIME_NAME"], True)
        port2 = self.get_port(row["SECOND_NAME"], True)
        new_fiber = Fiber(port1, port2)

        if self._is_fiber_complete(new_fiber):
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

                if self._is_fiber_complete(f):
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


    @staticmethod
    def _is_fiber_complete(fiber: Fiber):
        """ Checks if the fiber connects two switches directly. """
        return ConnectivityDatabase._is_wrs_port(fiber.start) \
            and ConnectivityDatabase._is_wrs_port(fiber.end)


    @staticmethod
    def _is_wrs_port(port: Port):
        """ Checks if the port is a White Rabbit Switch port. """
        return port.name.upper().startswith(ConnectivityDatabase.SWITCH_PORT_PREFIX)


    def process(self):
        self._process_incomplete_fibers()
        super(ConnectivityDatabase, self).process()


class MAC(int):
    """ Class representing a MAC address. """
    def __new__(cls, mac, *args, **kwargs):
        if isinstance(mac, str):
            mac = MAC._sanitize_mac(mac)
            mac = int(mac, 16)

        return super(MAC, cls).__new__(cls, mac)


    def __str__(self):
        return '{0:012x}'.format(self)


    @staticmethod
    def _sanitize_mac(mac):
        """ Removes colons and hyphens from the MAC address. """
        sanitized = mac.replace(':', '').replace('-', '').lower()

        if len(sanitized) != 12:
            raise ValueError("Invalid MAC address")

        return sanitized


class ConnectivityMAC(Connectivity):
    """ Class generating connectivity information basing on MAC addresses data. """
    def __init__(self, top_switch, all_connections=False):
        super(ConnectivityMAC, self).__init__(top_switch, all_connections)


    def get_port(self, mac, create=False):
        # use str version of MAC address to find ports, it is easier to read
        return Connectivity.get_port(self, str(mac), create)


    def get_switch_by_mac(self, mac):
        port = self.get_port(mac)

        if port is None or not port.device:
            return None

        try:
            return self._devices[port.device]
        except KeyError:
            return None


    def _build_mac_db(self):
        """ Scans all WR switches to build a MAC address database. """
        for entry in ccda.wr_switches():
            switch_name = entry["name"]

            try:
                switch = self.get_device(switch_name)
                self._process_switch_ports(switch)
            except ConnectionError:
                logger.warning(f"Could not connect to {switch_name}")


    def _get_mac1(self, switch):
        """ Returns the MAC address of the first SFP port. """
        raise NotImplementedError


    def _get_peer_mac(self, switch, port):
        """ Returns the MAC address of the peer connected to the switch port. """
        raise NotImplementedError


    def _port_range(self, switch):
        """ Returns the number of SFP ports on the switch. """
        raise NotImplementedError


    def _process_switch_ports(self, switch: Switch):
        # Get the MAC address of the first SFP port (remaining SFP ports have consecutive addresses)
        mac1 = self._get_mac1(switch)

        if mac1 is None:
            logger.warning(f"No MAC address found for {switch.name}, skipping")
            return

        # Bind all MACs to the switch
        for p in self._port_range(switch):
            # Compute the port MAC address
            mac = ConnectivityMAC._compute_mac(mac1, p)

            # Register the port, to create connections
            if self.get_port(mac):
                logger.warning(f"MAC {mac} already registered to {self.get_switch_by_mac(mac).name}, skipping {switch.name}")
                return

            new_port = self.get_port(mac, True)
            new_port.label = p             # Port number, indexed from 1
            new_port.device = switch.name
            switch.add_port(new_port)
            logger.debug(f"MAC DB: {mac} -> {switch.name}")


    def _process_switch_connections(self, switch: Switch):
        logger.debug(f"Processing {switch.name}")

        # Get the first SFP port MAC address as an integer (to compute MAC addresses of the remaining ports)
        my_mac1 = self._get_mac1(switch)

        # Check what is connected to each SFP port...
        for p in self._port_range(switch):
            peer_mac = self._get_peer_mac(switch, p)

            if peer_mac == None:
                continue        # nothing connected to the port

            my_mac = ConnectivityMAC._compute_mac(my_mac1, p)
            my_port = self.get_port(my_mac)
            peer_port = self.get_port(peer_mac)

            if peer_port is None:
                logger.warning(f"Unknown MAC address {peer_mac} on {my_port}")
                continue

            self._add_fiber(Fiber(my_port, peer_port))
            logger.debug(f"   {my_port} <-> {peer_port}")


    def process(self):
        # TODO instead of clearing, it would be better to track changes and update only what is needed
        self.clear()
        self._build_mac_db()

        # Add fibers basing on the LLDP data
        for s in self._devices.values():
            try:
                self._process_switch_connections(s)
            except ConnectionError:
                logger.warning(f"Could not process {s.name}")

        super(ConnectivityMAC, self).process()


    @staticmethod
    def _compute_mac(mac1_num, port_index):
        """ Calculates MAC address using first SFP port MAC address and the requested SFP port number (indexed from 1). """
        return MAC(mac1_num + port_index - 1)


class ConnectivityPTP(ConnectivityMAC):
    """ Class generating connectivity information from Icinga (which is currently based on PTP/SNMP). """
    def __init__(self, top_switch, all_connections=False):
        super(ConnectivityPTP, self).__init__(top_switch, all_connections)


    def get_device(self, name):
        # Nearly the same as Connectivity.get_device()
        # but this one creates Switch objects instead of Device
        name = name.lower()

        if name not in self._devices:
            self._devices[name] = Switch(name)

        return self._devices[name]


    def _build_mac_db(self):
        """ Scans all WR switches to build a MAC address database. """
        for entry in ccda.wr_switches():
            switch_name = entry["name"]

            try:
                switch = self.get_device(switch_name)
                self._process_switch_ports(switch)
            except ConnectionError:
                logger.warning(f"Could not connect to {switch_name}")


    def _get_mac1(self, switch):
        """ Returns the MAC address of the first SFP port. """
        return switch.snmp.sfp_port_mac(1)


    def _get_peer_mac(self, switch, port):
        return switch.snmp.sfp_port_peer_mac(port)


    def _port_count(self, switch):
        return switch.snmp.sfp_port_range()


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
    parser.add_argument("--top-switch", default=config.WR_GRANDMASTER,
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

    # Save the graph data
    conn.save_to_json(args.output)
