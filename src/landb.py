#!/usr/bin/env python3
import requests
import json
import sys


class LanDB:
    API_URL = "https://landb.cern.ch/api/v1"


    def __init__(self, auth_token):
        self._auth_token = auth_token


    def find_by_mac(self, mac):
        mac = mac.replace(':', '-')
        headers = { "Authorization": f"Bearer {self._auth_token}" }
        url = f"{LanDB.API_URL}/devices?_offset=0&_limit=1&macAddresses.any.eq={mac}"
        response = requests.get(url, headers=headers)
        return json.loads(response.text)
    

if __name__ == "__main__":
    # see https://auth.docs.cern.ch/user-documentation/oidc/api-access/ about getting the Bearer token
    import pprint
    if len(sys.argv) != 2:
        print(f'usage {sys.argv[0]} <bearer token>')
        exit(1)

    token = sys.argv[1]
    landb = LanDB(token)
    pprint.pprint(landb.find_by_mac("00-1B-C5-09-00-CD"))