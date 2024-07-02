#!/usr/bin/env python3
import json
import urllib.parse
import requests

CCDA_API_URL = 'https://ccda.cern.ch:8900/api'
CCDA_WR_SWITCH_TYPE = 'CTDW'

# TODO StrEnum is available starting from python 3.11
class Operation:
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    IN = "=in="
    OUT = "=out="
    AND = ";"
    OR = ","


class Query:
    """ Class used to construct complex queries for searching CCDB. """
    def __init__(self, left, operation: Operation, right):
        if operation in (Operation.IN, Operation.OUT) and not isinstance(right, tuple):
            raise TypeError

        self._left = left
        self._op = operation
        self._right = right

    @staticmethod
    def from_dict_and(dictionary: dict):
        """ Creates a query where all dictionary keys must equal their values (combined with 'and') """
        ret = None

        for k, v in dictionary.items():
            if ret is None:
                ret = Query(k, Operation.EQUAL, v)
            else:
                ret = ret & Query(k, Operation.EQUAL, v)

        return ret

    @staticmethod
    def from_dict_or(dictionary: dict):
        """ Creates a query where one of the dictionary keys must equal its value (combined with 'or') """
        ret = None

        for k, v in dictionary.items():
            if ret is None:
                ret = Query(k, Operation.EQUAL, v)
            else:
                ret = ret | Query(k, Operation.EQUAL, v)

        return ret

    def __and__(self, other):
        if not isinstance(other, Query):
            raise TypeError("Argument must be an instance of Query")

        return Query(self, Operation.AND, other)

    def __or__(self, other):
        if not isinstance(other, Query):
            raise TypeError("Argument must be an instance of Query")

        return Query(self, Operation.OR, other)

    def __str__(self):
        ret = str(self._left)
        ret += str(self._op)

        if isinstance(self._right, str):
            ret += f"'{self._right}'"
        else:
            ret += str(self._right)

        return ret


def crate_by_label(label):
    """ Finds a specific crate in CCDB (using 'label' field). """
    response = requests.get(f'{CCDA_API_URL}/crates/search?query=label%3D%3D{label}', verify=False)
    data = json.loads(response.text)

    if data['totalElements'] > 1:
        raise RuntimeError('Too many results')

    return data['content'][0]


def computer_by_name(name):
    """ Finds a specific computer in CCDB (using 'name' field). """
    response = requests.get(f'{CCDA_API_URL}/computers/{name}', verify=False)
    return json.loads(response.text)


def wr_switches(criteria=None):
    """
      Finds White Rabbit switches in CCDB.
      One can narrow the search results by passing a dictionary with search criteria
      (e.g. {'location': '774/R-051'})
    """
    if isinstance(criteria, dict):
        criteria['type'] = CCDA_WR_SWITCH_TYPE
        query = Query.from_dict_and(criteria)
    elif isinstance(criteria, Query):
        query = criteria and Query('type', Operation.EQUAL, CCDA_WR_SWITCH_TYPE)
    elif criteria is None:
        query = Query('type', Operation.EQUAL, CCDA_WR_SWITCH_TYPE)

    query = urllib.parse.quote(str(query))  # make it URL-friendly
    response = requests.get(f'{CCDA_API_URL}/computers/search?query={query}', verify=False)
    resp_dict = json.loads(response.text)
    return resp_dict['content']


if __name__ == "__main__":
    import pprint
    assert(str(Query("name", Operation.EQUAL, "ctdwa-774-cins3")) == "name=='ctdwa-774-cins3'")
    assert(str(Query("type", Operation.NOT_EQUAL, "CTDW")) == "type!='CTDW'")
    assert(str(Query("count", Operation.GREATER_THAN, 42)) == "count>42")
    assert(str(Query("number", Operation.LESS_THAN_EQUAL, 2)) == "number<=2")
    assert(str(Query("name", Operation.IN, ('ctdw-774-cins4', 'ctdw-774-cins5', 'ctdw-774-cins6'))) == "name=in=('ctdw-774-cins4', 'ctdw-774-cins5', 'ctdw-774-cins6')")

    assert(str(Query.from_dict_and({"field1": 5, "field2": "test", "field3": 0})) == "field1==5;field2=='test';field3==0")
    assert(str(Query.from_dict_or({"field1": "value", "field2": "test2", "field3": 9})) == "field1=='value',field2=='test2',field3==9")

    pprint.pprint(computer_by_name('cfu-864-alab34'))
    pprint.pprint(wr_switches({'location': '774/R-051'}))