#!/bin/bash
set -x
source env/bin/activate
python run_heuristics.py
deactivate