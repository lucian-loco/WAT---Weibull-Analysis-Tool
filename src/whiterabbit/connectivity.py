#!/usr/bin/env python3
import sys
import os.path
# add path to import modules from the directory above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import json
import logging
import argparse
import db_hitdata
import csv

from fiber import Fiber

logging.basicConfig(format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO do not hardcode it here!
WR_GRANDMASTER = "CTDW-CCR-CTNLJM1"


def id_generator(start=1):
    number = start
    while True:
        yield number
        number += 1


class Port:
    SWITCH_PORT_PREFIX = "CTDNT"

    def __init__(self, name, label=None, device=None, mac=None):
        self.name = name                # as in LayoutDB, e.g. CTDNT.168.BC
        self.label = label              # normally label is the port number, but not always
        self.device = device            # parent WR device name
        self.mac = mac.replace('-', ':').upper() if mac else None


    @property
    def is_wrs_port(self):
        return self.name.upper().startswith(Port.SWITCH_PORT_PREFIX)


    def __eq__(self, other):
        return self.name == other.name and self.label == other.label and self.device == other.device


    def __hash__(self):
        return hash(self.name)


    def __repr__(self):
        if isinstance(self.label, int):
            return "Port {0:02}/{1} ({2})".format(self.label, self.device, self.name)
        else:
            return "Port {0}/{1} ({2})".format(self.label, self.device, self.name)


class Device:
    SWITCH_NAME_PREFIX = "CTDW"

    _id_gen = id_generator(1)

    def __init__(self, name):
        assert(name is not None)
        self.name = name
        self.master = None
        self.id = None
        self.layer = 0
        self.ports = {}


    def is_switch(self):
        return self.name.upper().startswith(Device.SWITCH_NAME_PREFIX)


    def add_port(self, port):
        if port.name not in self.ports:
            self.ports[port.name] = port
        else:
            assert(port == self.ports[port.name])


    def sorted_ports(self):
        return sorted(self.ports.values(), key=lambda x: x.label)


    def assign_id(self):
        assert(self.id is None)
        self.id = next(Device._id_gen)


    def __repr__(self):
        return "Device {0}".format(self.name)


    def __lt__(self, other):
        return self.name < other.name




class Connectivity:
    def __init__(self, all_connections=False):
        self._all_connections = all_connections

        self._devices = {}
        self._ports = {}
        self._port_connections = {}
        self._fibers = []
        self._incomplete_fibers = []

        # visjs data
        self.edges = []
        self.nodes = []


    @staticmethod
    def from_csv(filename, *args, **kwargs):
        instance = Connectivity(*args, **kwargs)

        with open(filename) as f:
            reader = csv.DictReader(f)

            for row in reader:
                instance._process_row(row)

            instance.process()

        return instance


    @staticmethod
    def from_db(*args, **kwargs):
        # Note that there are more column available in the view,
        # but they are not used by the script
        columns = ("PRIME_NAME", "PRIME_LABEL", "PRIME_PARENT_NAME",
                "PRIME_PARENT_LABEL", "SECOND_NAME", "SECOND_LABEL",
                "SECOND_PARENT_NAME", "SECOND_PARENT_LABEL")

        cursor = db_hitdata.get_cursor()
        sql_query = "SELECT {0} FROM controls_wr_fibres_v".format(", ".join(columns))
        result = cursor.execute(sql_query)

        instance = Connectivity()

        for row in result:
            # Build a dictionary with columns as the keys
            row_annotated = {columns[i]: row[i] for i in range(len(columns))}
            instance._process_row(row_annotated)

        instance.process()
        return instance
    

    @staticmethod
    def from_snmp(*args, **kwargs):
        raise NotImplemented


    def get_device(self, name):
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


    def _add_complete_fiber(self, fiber):
        self._fibers.append(fiber)
        self._port_connections[fiber.start] = fiber.end
        self._port_connections[fiber.end] = fiber.start


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
            self._add_complete_fiber(new_fiber)
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
                    self._add_complete_fiber(f)
                else:
                    self._incomplete_fibers.append(f)

                return True

        return False


    def _handle_incomplete_fibers(self):
        merged_sth = True

        while merged_sth:
            merged_sth = False

            for f in self._incomplete_fibers:
                if self._try_to_merge_fiber(f):
                    merged_sth = True
                    # Restart the loop, _incomplete_fibers list has been modified
                    break   


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
                    continue

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
        self._handle_incomplete_fibers()

        grandmaster = self._devices[WR_GRANDMASTER]

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
            for name, device in self._devices.items():
                if device.master:
                    master = self._devices[device.master]
                    self.edges.append({ "from": device.id, "to": master.id })   # TODO description?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates WR connectivity information (JSON format)")
    
    parser.add_argument('--source', '-s', choices=('layoutdb', 'snmp', 'csv'), required=True,
            help='Data source.')
    parser.add_argument('--output', '-o', type=str, default='connectivity.json',
            help='Output file name.')
    parser.add_argument('--all', action='store_true',
            help='Include all connections (may create a non-tree structure).')
    args = parser.parse_args()

    if args.source == 'layoutdb':  
        conn = Connectivity.from_db(all_connections=args.all)
    elif args.source == 'snmp':
        conn = Connectivity.from_snmp(all_connections=args.all)
    elif args.source == 'csv':
        conn = Connectivity.from_csv("connections.csv", all_connections=args.all)

    logger.info("Incomplete fibers: {0}".format(len(conn._incomplete_fibers)))
    logger.info("Fibers: {0} Devices: {1}".format(len(conn._fibers), len(conn._devices)))

    #for name, device in conn._devices.items():
    #    print("> {0}".format(name))
    #    for port_number, port_data in device._ports.items():
    #        print(port_data)

    # Save graph data
    with open(args.output, "w") as output:
        json.dump({"edges": conn.edges, "nodes": conn.nodes}, output)
