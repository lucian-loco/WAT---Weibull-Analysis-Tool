#!/usr/bin/env python3
import logging
from getpass import getpass, getuser
from pathlib import Path
import socket
import suds
from suds.client import Client
from suds.xsd.doctor import Import
from suds.xsd.doctor import ImportDoctor


class ErrorNetOpsAuthentication(Exception):
    def __init__(self, msg):
        super().__init__("NetOps authentication failure: {}".format(msg))


class ErrorNetOps(Exception):
    def __init__(self, msg):
        super().__init__("NetOps failure: {}".format(msg))


class NetOps(object):
    def __init__(self, user=None, local_auth=True, dry_run=False):
        """
        Initialize a NetOps class
        """
        self.dry_run = dry_run
        wsdl = "https://network.cern.ch/sc/soap/soap.fcgi?v=6&WSDL"
        imp = Import('http://schemas.xmlsoap.org/soap/encoding/')
        doc = ImportDoctor(imp)
        self.client = Client(wsdl, doctor=doc)
        self.client.set_options(cache=None)

        self.netops = self.client.service
        token = self._get_token(user, local_auth)
        token_el = suds.sax.element.Element('token').setText(token)
        header = suds.sax.element.Element('Auth').insert(token_el)
        self.client.set_options(soapheaders=header)

    def _get_token(self, user, local_auth):
        if local_auth:
            return self.__get_token_local()
        else:
            return self.__get_token_user(user)

    def __get_token_local(self):
        """Get a LANDB token using local machine token

        Exceptions:
        ErrorNetOps: when the returnd token is invalid

        Returns:
        str: a token
        """
        authorized_servers = ["cs-ccr-feop.cern.ch"]
        if socket.gethostname() not in authorized_servers:
            raise ErrorNetOpsAuthentication("not authorized")

        with os.popen("super nettoken | cut -d= -f2- | cut -d\\; -f1") as proc:
            token = proc.read()
        if len(token) == 0:
            raise ErrorNetOpsAuthentication("failed to get token")
        return token

    def __get_token_user(self, user=None):
        """Get a LANDB token using user's credential

        Parameters:
        users: username for the authentication. By default it is None.

        Exceptions:
        ErrorNetOps: when the returnd token is invalid

        Returns:
        str: a token
        """
        tokenfile = Path('landbtoken')
        token = ""
        if tokenfile.exists():
            token = tokenfile.read_text()
            if len(token) == 0:
                raise ErrorNetOpsAuthentication("user authentication failed")
            return token

        if user is None:
            user = getuser()
        password = getpass("password:")

        try:
            token = self.netops.getAuthToken(user, password, "CERN")
        except suds.WebFault as e:
            if "WRONGLOGPASSW" in e.fault.faultstring:
                raise ErrorNetOpsAuthentication(e.fault.faultstring)
        if len(token) == 0:
            raise ErrorNetOpsAuthentication("authentication succeded but no token was returned")
        tokenfile.write_text(token)
        return token

    def getDeviceInfo(self, hostname):
        logging.debug("NetOps get computer info for hostname {}".format(hostname))
        try:
            return dict(self.netops.getDeviceInfo(hostname))
        except suds.WebFault as e:
            raise ErrorNetOps(e.fault.faultstring)

    def getBOOTPInfo(self, hostname):
        logging.debug("NetOps get DHCP boot options for {}".format(hostname))

        try:
            resp = self.netops.getBOOTPInfo(hostname)
            if not resp:
                logging.warning("No BOOTP configuration for {}".format(hostname))
                return {}
            info = dict(resp)
        except suds.WebFault as e:
            raise ErrorNetOps(e.fault.faultstring)
        return info

    def getInfo(self, hostname):
        device = self.getDeviceInfo(hostname)
        bootp = self.getBOOTPInfo(hostname)
        return {**device, **bootp}

    def searchDevice(self, criteria):
        logging.debug("NetOps search for computer names using multiple search criteria {}".format(criteria))
        assert(isinstance(criteria, dict))

        try:
            query = self.client.factory.create('ns1:DeviceSearch')
            # Workaround: by default there is another structure in 'Location',
            # but it is considered invalid by the server
            query.Location = None

            for k in criteria.keys():
                if k not in query.__dict__:
                    raise ErrorNetOps(f'Invalid search criterion: {k}')

            query.__dict__.update(criteria)
            return self.netops.searchDevice(query)
        except suds.WebFault as e:
            raise ErrorNetOps(e.fault.faultstring)

    def getDeviceInfoByMAC(self, mac):
        return self.searchDevice({'HardwareAddress': mac})

    def deviceSetBOOTPInfo(self, hostname, server, path):
        if not server.endswith(".cern.ch"):
            server += ".cern.ch"
        if not path.startswith("/"):
            path = f"/{path}"
        logging.debug(f"NetOps set hostname: {hostname}, next-server: {server}, filename: {path}")
        try:
            if self.dry_run:
                return None
            return self.netops.deviceSetBOOTPInfo(hostname, server, path)
        except suds.WebFault as e:
            raise ErrorNetOps(e.fault.faultstring)


if __name__ == "__main__":
    netops = NetOps(local_auth=False)
    print(netops.searchDevice({'Name': 'ctdwa%'}))
    #print(netops.getDeviceByMAC('00:1B:C5:09:00:CD'))
    #print(netops.getDeviceInfo('ctdwa-774-cins3'))
