#!/usr/bin/env python3
from enum import IntEnum
from devices import Device
import easysnmp
import sys
import os.path
# add path to import modules from the directory above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import ccda

class PortMode(IntEnum):
    UNKNOWN = 0
    MASTER = 1
    SLAVE = 2
    NON_WR = 3
    AUTO = 4
    NONE = 5

# TODO use getbulk?

class Switch(Device):
    def __init__(self, name):
        super(Switch, self).__init__(name)
        self._snmp = easysnmp.Session(hostname=name, community='public', version=2, retries=1, timeout=0.5)
        self._port_min = 1
        self._port_max = 18
        self._ccda_data = None          # cache for static data fetched using CCDA


    def port_check(func):
        def _check_func(self, index):
            if index < self._port_min or index > self._port_max:
                raise RuntimeError(f'Port number must be in range [{self._port_min}:{self._port_max}]')

            return func(self, index)

        return _check_func


    def ccda_data(func):
        def _query_if_needed(self):
            try:
                if self._ccda_data is None:
                    self._ccda_data = ccda.computer_by_name(self.name)
            except:
                return None

            return func(self)

        return _query_if_needed


    def _get_snmp(self, oid):
        try:
            response = self._snmp.get(oid)
        except easysnmp.exceptions.EasySNMPTimeoutError as e:
            exception = str(e)
            response = None

        if response is None:
            raise ConnectionError(exception) 

        if response.snmp_type in ('TICKS', 'INTEGER', 'Counter64'):
            return int(response.value)
        elif response.snmp_type == 'OCTETSTR':
            hex_values = ['{0:02x}'.format(ord(c)) for c in response.value]
            return ''.join(hex_values)

        return response.value


    # CCDA data
    @property
    @ccda_data
    def description(self): return self._ccda_data['description']

    @property
    @ccda_data
    def location(self): return self._ccda_data['location']

    @property
    @ccda_data
    def rack(self): return self._ccda_data['rack']

    @property
    @ccda_data
    def ccde_link(self): return 'https://ccde.cern.ch/hardware/computers/{0}'.format(self._ccda_data['id'])


    # General properties
    @property
    def system_description(self): return self._get_snmp('.1.3.6.1.2.1.1.1.0')

    @property
    def uptime(self): return self._get_snmp('.1.3.6.1.2.1.1.3.0')

    @property
    def hostname(self): return self._get_snmp('.1.3.6.1.2.1.1.5.0')


    # Status
    @property
    def status_main(self): return self._get_snmp('.1.3.6.1.4.1.96.100.6.1.1.0') == 1

    @property
    def status_os(self): return self._get_snmp('.1.3.6.1.4.1.96.100.6.1.2.0') == 1

    @property
    def status_timing(self): return self._get_snmp('.1.3.6.1.4.1.96.100.6.1.3.0') == 1

    @property
    def status_networking(self): return self._get_snmp('.1.3.6.1.4.1.96.100.6.1.3.0') == 1


    # Management port
    @property
    def mgmt_port_mac(self): return self._get_snmp('.1.3.6.1.2.1.2.2.1.6.2')


    # SFP ports
    @port_check
    def sfp_port_mac(self, index):
        return self._get_snmp(f'.1.3.6.1.2.1.2.2.1.6.{index+2}')

    @port_check
    def sfp_port_link_up(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.6.1.3.{index}') == 2

    @port_check
    def sfp_port_mode(self, index):
        mode = self._get_snmp(f'.1.3.6.1.4.1.96.100.7.6.1.4.{index}')

        try:
            return PortMode(mode)
        except ValueError:
            return PortMode.UNKNOWN

    @port_check
    def sfp_port_peer_mac(self, index):
        mac = self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.22.{index}.1')

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
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.23.{index}.1')

    @port_check
    def sfp_port_ptp_status_ok(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.26.{index}.1') == 1


if __name__ == "__main__":
    switch = Switch('ctdwa-774-cins3')
    print(switch.description)
    print(switch.mgmt_port_mac)

    for p in range(1, 19):
        print(switch.sfp_port_peer_mac(p))
