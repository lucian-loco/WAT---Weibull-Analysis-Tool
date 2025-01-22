#!/usr/bin/env python3
import sys
import os.path
# add path to import modules from the directory above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import easysnmp
from enum import Enum, IntEnum
from functools import wraps
import logging
logger = logging.getLogger(__name__)

from devices import Device
from cache import *
import ccda
import icinga

# TODO see what happens when a switch was off and now turned on, or reverse

class PortMode(IntEnum):
    """ Possible port configurations """
    UNKNOWN = 0
    MASTER = 1
    SLAVE = 2
    NON_WR = 3
    AUTO = 4
    NONE = 5


class PtpState(IntEnum):
    """ Possible PTP states """
    NA = 0
    INITIALIZING = 1
    FAULTY = 2
    DISABLED = 3
    LISTENING = 4
    PRE_MASTER = 5
    MASTER = 6
    PASSIVE = 7
    UNCALIBRATED = 8
    SLAVE = 9


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


class CacheLLDP(ExpiringCacheMixin):
    """
      Caches the LLDP data to prevent too frequent SNMP queries.
      The values are updated every cache_timeout seconds.
      One cannot use the standard CacheSNMP class, because processing LLDP data requires
      an SNMP walk (OIDs are not fixed/known in advance).
    """
    def __init__(self, session, cache_timeout):
        super(CacheLLDP, self).__init__(cache_timeout)
        self._session = session
        self._cache = {}

    @ExpiringCacheMixin.decorate()
    def get(self, port):
        return self._cache.get(port, None)

    def _update_cache(self):
        logger.debug(f'Updating SNMP cache for {self._session.hostname}')

        self._cache = {}
        lldp_connections = self._session.walk('.1.0.8802.1.1.2.1.4.1.1.7')

        for item in lldp_connections:
            assert(item.snmp_type == 'OCTETSTR')
            port = int(item.oid.split('.')[-2]) - 2

            if port == 0:   # skip the management port
                continue

            peer_mac = ''.join([f'{ord(x):02x}' for x in item.value])
            assert(port not in self._cache)
            self._cache[port] = peer_mac


class SwitchSNMP:
    class Source(Enum):
        """ Enumeration of possible sources for the peer MAC address. """
        AUTO = 0 # LLDP preferred, PTP as a fallback
        PTP = 1
        LLDP = 2

    def __init__(self, name, data_source=Source.AUTO):
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
            *(f'iso.3.6.1.4.1.96.100.7.8.1.7.1.{i}' for i in self.sfp_port_range()),  # PTP instance state
        ]

        if data_source == SwitchSNMP.Source.AUTO:    # automatic selection, LLDP preferred if available
            try:
                # Check if lldpd is started on the switch
                lldp_started = self._get_snmp('.1.3.6.1.4.1.96.100.7.2.9.0') # WR-SWITCH-MIB::wrsStartCntLldpd.0
                if int(lldp_started) > 0:
                    logger.debug(f'Checking LLDP on {name}: enabled')
                    self._use_lldp = True
                else:
                    logger.debug(f'Checking LLDP on {name}: disabled')
                    self._use_lldp = False
            except ConnectionError:
                logger.info(f'Could not determine if LLDP is enabled on {name}, assuming disabled')
                self._use_lldp = False
        else:
            self._use_lldp = data_source == SwitchSNMP.Source.LLDP

        # Select OIDs used for connectivity checking (either PTP or LLDP)
        if self._use_lldp:
            self._lldp_cache = CacheLLDP(self._session, 60)
        else:
            self._oids_variable.extend([
                *(f'iso.3.6.1.4.1.96.100.7.8.1.22.{i}.1' for i in self.sfp_port_range()), # SFP port peerMAC
                *(f'iso.3.6.1.4.1.96.100.7.8.1.23.{i}.1' for i in self.sfp_port_range()), # SFP port peerVID
                *(f'iso.3.6.1.4.1.96.100.7.8.1.26.{i}.1' for i in self.sfp_port_range())  # SFP port ptpStatusOK
            ])

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
                logger.info(f'Querying non-cached OID: {oid}')
                response = self._session.get(oid)
            except easysnmp.exceptions.EasySNMPTimeoutError as e:
                raise ConnectionError from e

        if response is None:
            # TODO it means the host has not responded the last time,
            # the cache is empty, so maybe the request should be repeated
            raise ConnectionError

        # Response formatting
        snmp_type = response.snmp_type.lower()
        if snmp_type in ('ticks', 'integer', 'counter64'):
            return int(response.value)
        elif snmp_type == 'octetstr':
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


    # PTP
    @port_check
    def ptp_state(self, index):
        state = self._get_snmp(f'iso.3.6.1.4.1.96.100.7.8.1.7.{index}.1')

        try:
            return PtpState(state)
        except ValueError:
            return PtpState.NA


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
        if self._use_lldp:
            mac = self._lldp_cache.get(index)
        else:
            mac = self._get_snmp(f'iso.3.6.1.4.1.96.100.7.8.1.22.{index}.1')

            if mac == "000000000000" or mac.startswith("NOSUCH"):   # nothing was ever connected here
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


class SwitchIcinga(ExpiringCacheMixin):
    # Icinga API connection (do not use directly, use _api() method)
    __icinga_api_handle = None

    def __init__(self, name):
        super(SwitchIcinga, self).__init__(5 * 60)
        self.name = name
        self._data = None


    @classmethod
    def _api(cls):
        """ Returns the Icinga API object. """
        if cls.__icinga_api_handle is None:
            user = os.environ['ICINGA_USER']
            password = os.environ['ICINGA_PASS']
            hostname = os.environ['ICINGA_HOST']
            cls.__icinga_api_handle = icinga.IcingaAPI(user, password, hostname)

        return cls.__icinga_api_handle


    def _update_cache(self):
        logger.debug(f'Updating Icinga cache for {self.name}')
        self._data = {}

        # Get all services data
        q = icinga.ObjectQuery(icinga.ObjectType.SERVICE)
        q.filter_equal('host.name', self.name)
        q.filter_match('service.name', 'WRS*Service')
        q.add_attribute('last_check_result')
        result = SwitchIcinga._api().execute_query(q)

        # result is an array of dictionaries, one for each service
        for r in result:
            if r['attrs']['last_check_result']['state'] == icinga.ServiceState.UNKNOWN:
                logger.warning(f"Icinga service {r['name']} is in UNKNOWN state")
                continue

            service_name = r['name'].split('!')[1]
            self._data[service_name] = icinga.IcingaAPI.parse_performance_data(r['attrs']['last_check_result']['performance_data'])


    @property
    @ExpiringCacheMixin.decorate(4*60)
    def status(self):
        """ Returns the general status. """
        return self._data['WRSStatusService']

    @property
    @ExpiringCacheMixin.decorate(4*60)
    def system(self):
        """ Returns the operating system status. """
        return self._data['WRSSystemService']

    @property
    @ExpiringCacheMixin.decorate(4*60)
    def connectivity(self):
        """ Returns the connectivity status. """
        return self._data['WRSConnectivityService']

    @property
    @ExpiringCacheMixin.decorate(4*60)
    def temperature(self):
        """ Returns the temperature status. """
        return self._data['WRSTemperatureService']

    @property
    @ExpiringCacheMixin.decorate(30*60)
    def version(self):
        """ Returns the version status. """
        return self._data['WRSVersionService']

    @property
    @ExpiringCacheMixin.decorate(4*60)
    def network(self):
        """ Returns the network status. """
        return self._data['WRSNetworkService']

    @property
    @ExpiringCacheMixin.decorate(4*60)
    def timing(self):
        """ Returns the timing status. """
        return self._data['WRSTimingService']

    @property
    @ExpiringCacheMixin.decorate(60*60)
    def lifetime(self):
        """ Returns the lifetime status. """
        return self._data['WRSLifeTimeService']


class Switch(Device):
    """ Represents a White Rabbit switch and provides access to its data. """

    def __init__(self, name, snmp_source=SwitchSNMP.Source.AUTO):
        super(Switch, self).__init__(name)
        self.ccda = SwitchCCDA(name)

        try:
            self.snmp = SwitchSNMP(name, data_source=snmp_source)
        # Convert SNMP exceptions to ConnectionError
        except easysnmp.exceptions.EasySNMPConnectionError as e:
            raise ConnectionError from e
        except easysnmp.exceptions.EasySNMPTimeoutError as e:
            raise ConnectionError from e

        self.icinga = SwitchIcinga(name)


if __name__ == "__main__":
    import time
    import pprint
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

    # Icinga test
    pprint.pprint(switch.icinga.connectivity)
    pprint.pprint(switch.icinga.temperature)
    pprint.pprint(switch.icinga.status)