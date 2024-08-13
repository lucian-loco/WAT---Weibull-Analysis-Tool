#!/usr/bin/env python
import requests
import json

# TODO Python 3.10 does not have StrEnum
class ObjectType:
    """
    Represents the types of objects in Icinga.
    """
    HOST = 'host'
    SERVICE = 'service'
    HOSTGROUP = 'hostgroup'
    SERVICEGROUP = 'servicegroup'


class ServiceState:
    """
    Represents the states of a service in Icinga.
    """
    OK = 0
    WARNING = 1
    CRITICAL = 2
    UNKNOWN = 3


class HostState:
    """
    Represents the states of a host in Icinga.
    """
    UP = 0
    DOWN = 1
    UNREACHABLE = 2


class ObjectQuery:
    """
    Represents a query for retrieving objects data from Icinga.
    """

    def __init__(self, type, attrs=[], joins=[], all_joins=False, filter=""):
        """
        Initializes a new instance of the ObjectQuery class.

        Args:
            type (str): The type of objects to query.
            attrs (list, optional): The attributes to include in the query result. Defaults to an empty list.
            joins (list, optional): The joins to include in the query. Defaults to an empty list.
            filter (str, optional): The filter expression to apply to the query. Defaults to an empty string.
        """
        if all_joins and bool(joins):
            raise ValueError('Cannot specify both joins and all_joins')
        
        if not isinstance(attrs, list):
            raise ValueError('attrs must be a list')
        
        if not isinstance(joins, list): 
            raise ValueError('joins must be a list')
        
        self._type = type
        self._attrs = attrs
        self._joins = joins
        self._all_joins = all_joins
        self._filter = filter

    def make_url(self):
        """
        Generates the URL for the query.

        Returns:
            str: The URL for the query.
        """
        return f'/v1/objects/{self._type}s'

    def make_data(self):
        """
        Generates the data payload for the query.

        Returns:
            dict: The data payload for the query.
        """
        assert(not (bool(self._joins) and self._all_joins)) # these options are mutually exclusive
        data = {}
        if self._attrs:
            data['attrs'] = self._attrs
        if self._joins:
            data['joins'] = self._joins
        if self._all_joins:
            data['all_joins'] = 1
        if self._filter:
            data['filter'] = self._filter
        return data

    def __str__(self):
        """
        Returns a string representation of the ObjectQuery object.

        Returns:
            str: The string representation of the ObjectQuery object.
        """
        return json.dumps(self.to_dict())

    def add_attribute(self, attr: str):
        """
        Adds an attribute to include in the query result.

        Args:
            attr (str): The attribute to add.
        """
        self._attrs.append(attr)

    def add_join(self, join: str):
        """
        Adds a join to include in the query.

        Args:
            join (str): The join to add.
        """
        self._joins.append(join)

    def _add_filter(self, filter: str):
        """
        Adds a filter expression to the query.

        Args:
            filter (str): The filter expression to add.
        """
        if self._filter:
            self._filter += ' && ' + filter
        else:
            self._filter = filter

    def filter_match(self, field: str, pattern: str):
        """
        Adds a match filter expression to the query.

        Args:
            field (str): The field to match against.
            pattern (str): The pattern to match.
        """
        self._add_filter(f'match("{pattern}",{field})')

    def filter_equal(self, field: str, value):
        """
        Adds an equal filter expression to the query.

        Args:
            field (str): The field to compare.
            value: The value to compare against.
        """
        if isinstance(value, str):
            self._add_filter(f'{field}=="{value}"')
        else:
            self._add_filter(f'{field}=={str(value)}')

    def filter_not_equal(self, field: str, value):
        """
        Adds a not equal filter expression to the query.

        Args:
            field (str): The field to compare.
            value: The value to compare against.
        """
        if isinstance(value, str):
            self._add_filter(f'{field}!="{value}"')
        else:
            self._add_filter(f'{field}!={str(value)}')


class IcingaAPI:
    """
    A class allowing to query Icinga API.

    Args:
        user (str): The username for authentication.
        password (str): The password for authentication.
        url (str): The URL of the Icinga API.
        ssl_cert (optional): SSL certificate path. Defaults to False (do not verify).
    """

    def __init__(self, user, password, url, ssl_cert=False):
        self._user = user
        self._password = password
        self._url = url
        self._ssl_cert = ssl_cert


    def execute_query(self, object_query: ObjectQuery):
        headers = {
            'Accept': 'application/json',
            'X-HTTP-Method-Override': 'GET'
        }

        response = requests.get(self._url + object_query.make_url(),
                                 data=json.dumps(object_query.make_data()),
                                 headers=headers,
                                 auth=(self._user, self._password),
                                 verify=self._ssl_cert)

        if response.status_code != 200:
            raise RuntimeError(f'HTTP status code {response.status_code}')

        return response.json()['results']


    def get_host(self, host):
        q = ObjectQuery(ObjectType.HOST, filter=f'host.name=="{host}"')
        return self.execute_query(q)


    def get_service(self, host, service):
        q = ObjectQuery(ObjectType.SERVICE)
        q.filter_equal('host.name', host)
        q.filter_equal('service.name', service)
        return self.execute_query(q)


    def get_hostgroup(self, hostgroup):
        q = ObjectQuery(ObjectType.HOSTGROUP, filter=f'hostgroup.name=="{hostgroup}"')
        return self.execute_query(q)


    def get_servicegroup(self, servicegroup):
        q = ObjectQuery(ObjectType.SERVICEGROUP, filter=f'servicegroup.name=="{servicegroup}"')
        return self.execute_query(q)


if __name__ == '__main__':
    import pprint
    import os

    ICINGA_USER = os.environ['ICINGA_USER']
    ICINGA_PASS = os.environ['ICINGA_PASS']
    ICINGA_HOST = os.environ['ICINGA_HOST']
    icinga = IcingaAPI(ICINGA_USER, ICINGA_PASS, ICINGA_HOST)

    # Simple queries
    #pprint.pprint(icinga.get_host('ctdw-774-cptst1'))
    #pprint.pprint(icinga.get_host('ctdwa-774-cins3'))
    #pprint.pprint(icinga.get_service('ctdwa-774-cins3', 'WRSTemperatureService'))
    #pprint.pprint(icinga.get_servicegroup("WRSwitchServiceGroup"))
    #pprint.pprint(icinga.get_service('ctdwa-774-cins3', 'WRSTemperatureService'))

    # Complex query example
    # Find all services with name WRSTemperatureService on hosts starting with ctdw-774-cins..
    # and show their host name and IP address (using joins)
    q = ObjectQuery(ObjectType.SERVICE, joins=['host.name', 'host.address'])
    q.filter_equal('host.name', 'ctdw-774-cins4')
    q.filter_match('service.name', '*')
    # Show only the following attributes
    #q.add_attribute('last_check_result')
    #q.add_attribute('active')
    pprint.pprint(icinga.execute_query(q))