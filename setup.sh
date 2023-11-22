#!/bin/bash

python3 -m venv venv --prompt freebus
source venv/bin/activate
pip install -r dev_requirements.txt
pip install -r requirements.txt
pip install -e .
