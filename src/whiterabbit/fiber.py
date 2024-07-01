#!/usr/bin/env python3
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


    @property
    def is_complete(self):
        """ Checks whether this fiber connects two WR switches together (and not e.g. a patch panel) """
        return self.start.is_wrs_port and self.end.is_wrs_port


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
