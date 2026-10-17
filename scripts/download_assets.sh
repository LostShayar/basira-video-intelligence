#!/usr/bin/env bash
# Put sample videos in data/ (gitignored). Model weights download automatically
# the first time Ultralytics loads them.
#
# Use public clips whose licence allows your use, for example scenes from
# MOT17/MOT20, UA-DETRAC or VIRAT, or CC-licensed street footage.
# Rename them to match configs/cameras.yaml:
#   data/plaza_north.mp4
#   data/main_road.mp4
#   data/station.mp4
set -euo pipefail
mkdir -p data
echo "Place your sample videos in ./data and match the names in configs/cameras.yaml"
ls -la data
