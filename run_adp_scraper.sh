#!/bin/bash
# ADP Scraper Runner with proper environment
# Usage: bash run_adp_scraper.sh

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
python -u Load_Data/load_ADP_Employment_change.py
