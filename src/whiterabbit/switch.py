#!/usr/bin/env python3
from easysnmp import Session
from enum import IntEnum
import ccda

class PortMode:
    MASTER = 1
    SLAVE = 2
    UNKNOWN = 255

class Switch:
    def __init__(self, hostname):
        self._hostname = hostname
        self._snmp = Session(hostname=hostname, community='public', version=2)
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
                    self._ccda_data = ccda.computer_by_name(self._hostname)
            except:
                return None

            return func(self)

        return _query_if_needed


    def _get_snmp(self, oid):
        response = self._snmp.get(oid)

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
    def sfp_port_link(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.6.1.3.{index}') == 2

    @port_check
    def sfp_port_mode(self, index):
        mode = self._get_snmp(f'.1.3.6.1.4.1.96.100.7.6.1.4.{index}')

        # TODO check
        if mode == 1:
            return PortMode.MASTER
        elif mode == 2:
            return PortMode.SLAVE
        else:
            return PortMode.UNKNOWN

    @port_check
    def sfp_port_peer_mac(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.22.{index}.1')

    @port_check
    def sfp_port_peer_vid(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.23.{index}.1')

    @port_check
    def sfp_port_ptp_status(self, index):
        return self._get_snmp(f'.1.3.6.1.4.1.96.100.7.8.1.26.{index}.1') == 1


if __name__ == "__main__":
    switch = Switch('ctdwa-774-cins3')
    print(switch.description)
    print(switch.mgmt_port_mac)
    print(switch.sfp_port_mac(1))
    print(switch.sfp_port_peer_mac(3))
