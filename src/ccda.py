#!/usr/bin/env python3
import json
import requests

CCDA_API_URL='https://ccda.cern.ch:8900/api'


def crate_by_label(label):
    response = requests.get(f'{CCDA_API_URL}/crates/search?query=label%3D%3D{label}', verify=False)
    data = json.loads(response.text)

    if data['totalElements'] <= 0:
        raise RuntimeError('Invalid crate label')

    if data['totalElements'] > 1:
        raise RuntimeError('Too many results')

    return data['content'][0]


def computer_by_name(name):
    response = requests.get(f'{CCDA_API_URL}/computers/{name}', verify=False)
    return json.loads(response.text)