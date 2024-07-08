#!/usr/bin/env python3

class Port:
    """
    Class representing a White Rabbit switch port.
    """
    def __init__(self, name, label=None, device=None, mac=None):
        self.name = name                # as in LayoutDB, e.g. CTDNT.168.BC
        self.label = label              # normally label is the port number, but not always
        self.device = device            # parent WR device name
        self.mac = mac.replace('-', ':').upper() if mac else None


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
    """
    Class representing a network device
    (e.g. White Rabbit switch, White Rabbit node, etc.)
    """
    SWITCH_NAME_PREFIX = "ctdw"

    @staticmethod
    def id_generator(start=1):
        """ Generates unique device IDs """
        number = start
        while True:
            yield number
            number += 1

    _id_gen = id_generator(1)


    def __init__(self, name):
        assert(name is not None)
        self.name = name.lower()
        self.master = None
        self.id = None
        self.layer = 0
        self.ports = {}


    def is_switch(self):
        return self.name.startswith(Device.SWITCH_NAME_PREFIX)


    def add_port(self, port):
        if port.name not in self.ports:
            self.ports[port.name] = port
        else:
            assert(port == self.ports[port.name])


    def sorted_ports(self):
        """ Returns ports sorted by label. """
        return sorted(self.ports.values(), key=lambda x: x.label)


    def assign_id(self):
        assert(self.id is None)
        self.id = next(Device._id_gen)


    def __repr__(self):
        return "Device {0}".format(self.name)


    def __lt__(self, other):
        return self.name < other.name


class Fiber:
    """
    Class representing a fiber connection.
    Fibers may connect different kinds of nodes, e.g. WR nodes, WR switches, patch panels, etc.
    """
    def __init__(self, start, end):
        assert(start != end)
        self._nodes = [start, end]


    @property
    def start(self):
        return self._nodes[0]


    @property
    def end(self):
        return self._nodes[-1]


    def merge(self, other):
        """ Merges another cable segment to this one (other is not modified) """
        if self == other:
            raise RuntimeError("Cannot merge a fiber to itself")

        if self.start == other.start:   # self=[1,2,3] other=[1,4,5] => [5,4,1,2,3]
            self._nodes = [*other._nodes[::-1], *self._nodes[1:]]
        elif self.start == other.end:   # self=[1,2,3] other=[5,4,1] => [5,4,1,2,3]
            self._nodes = [*other._nodes, *self._nodes[1:]]
        elif self.end == other.start:   # self=[1,2,3] other=[3,4,5] => [1,2,3,4,5]
            self._nodes = [*self._nodes, *other._nodes[1:]]
        elif self.end == other.end:     # self=[1,2,3] other=[5,4,3] => [1,2,3,4,5]
            self._nodes = [*self._nodes[:-1], *other._nodes[::-1]]
        else:
            raise RuntimeError("Cannot merge fibers: {0} and {1}".format(self, other))

        assert(self.start != self.end)


    def can_merge(self, other):
        """ Checks if two fibers can be merged into a single connection """
        # Fibers can be merged only if they have a common end
        if self == other:
            return False

        return self.start == other.start or self.start == other.end or \
               self.end == other.end or self.end == other.end

    
    def __repr__(self):
        return "{0}-{1}".format(self.start, self.end)


    def __eq__(self, other):
        # TODO would it be enough to compare only start and end?
        # TODO should they be considered equal if they connect the same nodes, but in reverse direction?
        #      i.e. f1.start == f2.end and f1.end == f2.start
        return self._nodes == other._nodes