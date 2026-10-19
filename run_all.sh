#!/usr/bin/env bash
# Reproduces every number and chart: tests, then studies, then the report.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out
python3 -m pytest -q tests
python3 study.py
python3 report.py
