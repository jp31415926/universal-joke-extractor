#!/usr/bin/bash

rm -vrf __pycache__ .pytest_cache tests/__pycache__
rm -vf `find . -path "./.venv" -prune -o -name "*~" -print`
