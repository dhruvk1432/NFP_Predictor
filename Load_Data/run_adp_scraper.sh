#!/bin/bash
# ADP Scraper Runner with proper environment
# Usage: bash Load_Data/run_adp_scraper.sh (from project root)

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
python -u Load_Data/load_ADP_Employment_change.py
