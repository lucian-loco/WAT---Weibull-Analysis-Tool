#!/usr/bin/env python3
import sys
import os.path
# add path to import modules from the directory above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import easysnmp
from enum import IntEnum
from functools import wraps
import logging
logger = logging.getLogger(__name__)

from devices import Device
from cache import *
import ccda

# TODO see what happens when a switch was off and now turned on, or reverse

class PortMode(IntEnum):
    """ Possible port configurations """
    UNKNOWN = 0
    MASTER = 1
    SLAVE = 2
    NON_WR = 3
    AUTO = 4
    NONE = 5


class SwitchCCDA(ExpiringCacheMixin):
    def __init__(self, name):
        super(SwitchCCDA, self).__init__(4 * 60 * 60)   # 4 hours expiration period
        self.name = name
        self._data = None


    def _update_cache(self):
        logger.debug(f'Updating CCDA cache for {self.name}')
        self._data = ccda.computer_by_name(self.name)


    # CCDA data
    @property
    @ExpiringCacheMixin.decorate()
    def description(self): return self._data['description']

    @property
    @ExpiringCacheMixin.decorate()
    def location(self): return self._data['location']

    @property
    @ExpiringCacheMixin.decorate()
    def rack(self): return self._data['rack']

    @property
    @ExpiringCacheMixin.decorate()
    def ccde_link(self): return 'https://ccde.cern.ch/hardware/computers/{0}'.format(self._data['id'])


class CacheSNMP(ExpiringCacheMixin):
    """
      Caches the values of the OIDs to prevent too frequent SNMP queries.
      The values are updated every cache_timeout seconds.
    """
    def __init__(self, session, oids, cache_timeout):
        super(CacheSNMP, self).__init__(cache_timeout)
        assert all(oid.startswith('iso') for oid in oids), "All OIDs must start with 'iso'"
        self._session = session
        self._oids = oids
        self._cache = {}


    @ExpiringCacheMixin.decorate()
    def get(self, oid):
        assert(oid in self._oids)
        return self._cache[oid]


    def _update_cache(self):
        logger.debug(f'Updating SNMP cache for {self._session.hostname}')

        try:
            for r in self._session.get(self._oids):
                self._cache[r.oid] = r
        except easysnmp.exceptions.EasySNMPTimeoutError as e:
            # Invalidate the cache in case of a timeout
            for k in self._oids:
                self._cache[k] = None


class SwitchSNMP:
    def __init__(self, name):
        self.name = name
        self._port_min = 1
        self._port_max = 18     # TODO do not hardcode, get from SNMP?
        self._session = easysnmp.Session(hostname=name, community='public', version=2, retries=1, timeout=0.5)

        # List of OIDs which are not supposed to change often (unless you replace the switch, update firmware, etc.)
        self._oids_constant = [
            'iso.3.6.1.2.1.1.5.0',           # Hostname
            'iso.3.6.1.2.1.1.1.0',           # System description
            'iso.3.6.1.2.1.2.2.1.6.2',       # MAC address of the management port
            *(f'iso.3.6.1.2.1.2.2.1.6.{i+2}' for i in self.sfp_port_range()) # SFP port MAC addresses
        ]
        self._cache_constant = CacheSNMP(self._session, self._oids_constant, 4 * 60 * 60)   # 4 hours expiration period

        # List of default OIDs which are likely to change more often (or at least should be checked more often)
        self._oids_variable = [
            'iso.3.6.1.2.1.1.3.0',           # Uptime
            'iso.3.6.1.4.1.96.100.6.1.1.0',  # Main status
            'iso.3.6.1.4.1.96.100.6.1.2.0',  # Operating system status
            'iso.3.6.1.4.1.96.100.6.1.3.0',  # Timing status
            'iso.3.6.1.4.1.96.100.6.1.4.0',  # Networking status
            *(f'iso.3.6.1.4.1.96.100.7.6.1.3.{i}' for i in self.sfp_port_range()),  # SFP port linkUp
            *(f'iso.3.6.1.4.1.96.100.7.6.1.4.{i}' for i in self.sfp_port_range()),  # SFP port mode
            *(f'iso.3.6.1.4.1.96.100.7.8.1.22.{i}.1' for i in self.sfp_port_range()), # SFP port peerMAC
            *(f'iso.3.6.1.4.1.96.100.7.8.1.23.{i}.1' for i in self.sfp_port_range()), # SFP port peerVID
            *(f'iso.3.6.1.4.1.96.100.7.8.1.26.{i}.1' for i in self.sfp_port_range())  # SFP port ptpStatusOK
        ]
        self._cache_variable = CacheSNMP(self._session, self._oids_variable, 60)    # 1 minute expiration period


    def port_check(func):
        """ Decorator for checking if the port number is in the correct range. """
        @wraps(func)
        def wrapper(self, index):
            if index < self._port_min or index > self._port_max:
                raise RuntimeError(f'Port number must be in range [{self._port_min}:{self._port_max}]')

            return func(self, index)

        return wrapper


    def _get_snmp(self, oid):
        """ Returns the value of the OID. """
        if oid in self._oids_constant:
            response = self._cache_constant.get(oid)

        elif oid in self._oids_variable:
            response = self._cache_variable.get(oid)

        else:   # non-cached OIDs
            try:
                logger.warning(f'Querying non-cached OID: {oid}')
                response = self._session.get(oid)
            except easysnmp.exceptions.EasySNMPTimeoutError as e:
                raise ConnectionError from e

        if response is None:
            # TODO it means the host has not responded the last time,
            # the cache is empty, so maybe the request should be repeated
            raise ConnectionError

        # Response formatting
        if response.snmp_type in ('TICKS', 'INTEGER', 'Counter64'):
            return int(response.value)
        elif response.snmp_type == 'OCTETSTR':
            hex_values = ['{0:02x}'.format(ord(c)) for c in response.value]
            return ''.join(hex_values)

        return response.value


    def sfp_port_range(self):
        """ Returns a range of SFP port numbers. """
        return range(self._port_min, self._port_max + 1)


    def set_constant_expiration_period(self, expiration_period):
        """ Set the expiration period for the constant cache. """
        self._cache_constant.set_expiration_period(expiration_period)


    def set_variable_expiration_period(self, expiration_period):
        """ Set the expiration period for the variable cache. """
        self._cache_variable.set_expiration_period(expiration_period)


    # General properties
    @property
    def system_description(self): return self._get_snmp('iso.3.6.1.2.1.1.1.0')

    @property
    def uptime(self): return self._get_snmp('iso.3.6.1.2.1.1.3.0')

    @property
    def hostname(self): return self._get_snmp('iso.3.6.1.2.1.1.5.0')


    # Status
    @property
    def status_main(self): return self._get_snmp('iso.3.6.1.4.1.96.100.6.1.1.0') == 1

    @property
    def status_os(self): return self._get_snmp('iso.3.6.1.4.1.96.100.6.1.2.0') == 1

    @property
    def status_timing(self): return self._get_snmp('iso.3.6.1.4.1.96.100.6.1.3.0') == 1

    @property
    def status_networking(self): return self._get_snmp('iso.3.6.1.4.1.96.100.6.1.3.0') == 1


    # Management port
    @property
    def mgmt_port_mac(self): return self._get_snmp('iso.3.6.1.2.1.2.2.1.6.2')


    # SFP ports
    @port_check
    def sfp_port_mac(self, index):
        return self._get_snmp(f'iso.3.6.1.2.1.2.2.1.6.{index+2}')

    @port_check
    def sfp_port_link_up(self, index):
        return self._get_snmp(f'iso.3.6.1.4.1.96.100.7.6.1.3.{index}') == 2

    @port_check
    def sfp_port_mode(self, index):
        mode = self._get_snmp(f'iso.3.6.1.4.1.96.100.7.6.1.4.{index}')

        try:
            return PortMode(mode)
        except ValueError:
            return PortMode.UNKNOWN

    @port_check
    def sfp_port_peer_mac(self, index):
        mac = self._get_snmp(f'iso.3.6.1.4.1.96.100.7.8.1.22.{index}.1')

        if mac == "000000000000":   # nothing was ever connected here
            return None

        # Check if the port is really up
        # (it may happen that a device used to be connected there,
        # now it is disconnected but MAC is still preserved).
        if not self.sfp_port_link_up(index):
            return None

        return mac

    @port_check
    def sfp_port_peer_vid(self, index):
        return self._get_snmp(f'iso.3.6.1.4.1.96.100.7.8.1.23.{index}.1')

    @port_check
    def sfp_port_ptp_status_ok(self, index):
        return self._get_snmp(f'iso.3.6.1.4.1.96.100.7.8.1.26.{index}.1') == 1


class Switch(Device):
    def __init__(self, name):
        super(Switch, self).__init__(name)
        self.ccda = SwitchCCDA(name)
        self.snmp = SwitchSNMP(name)


if __name__ == "__main__":
    import time
    logger.setLevel(logging.DEBUG)
    switch = Switch('ctdwa-774-cins3')

    # CCDA test
    switch.ccda.set_expiration_period(1)
    print(switch.ccda.description)
    print(switch.ccda.rack)
    time.sleep(3)
    print(switch.ccda.location)

    # SNMP test
    switch.snmp.set_variable_expiration_period(2)
    uptime = switch.snmp.uptime
    print(switch.snmp.mgmt_port_mac)
    print(switch.snmp.hostname)
    assert(uptime == switch.snmp.uptime)    # uptime should not be refreshed yet
    time.sleep(3)
    print(switch.snmp.uptime)
    assert(uptime != switch.snmp.uptime)    # uptime should be refreshed now

    for p in range(1, 19):
        print('SFP{0} peer MAC: {1}'.format(p, switch.snmp.sfp_port_peer_mac(p)))