#!/usr/bin/env python3
import requests

API_TOKEN_URL = "https://auth.cern.ch/auth/realms/cern/api-access/token"

def get_bearer_token(client_id, client_secret, audience):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    data = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'audience': audience
    }

    response = requests.post(API_TOKEN_URL, headers=headers, data=data)

    if response.status_code != 200:
        raise RuntimeError('Could not acquire authentication token: ' + response.text)

    return response.json()


if __name__ == '__main__':
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description='Utility to acquire Bearer authentication token')
    parser.add_argument('--client_id', required=True, help='Client ID')
    parser.add_argument('--client_secret', required=True, help='Client secret')
    parser.add_argument('--audience', required=True, help='Target API')
    args = parser.parse_args()

    token = get_bearer_token(args.client_id, args.client_secret, args.audience)
    pprint.pprint(token)
