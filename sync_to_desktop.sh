#!/bin/bash
# Sync Lucifer-experimental to Windows desktop
# Usage: sync_to_desktop.sh [--checkpoints-only]

SRC="/home/suda/Lucifer-experimental"
DST="/mnt/c/Users/Sidon/Desktop/Lucifer"

mkdir -p "$DST"

if [ "$1" = "--checkpoints-only" ]; then
    # Sync only checkpoints, training log, and curriculum state
    rsync -a --delete "$SRC/Checkpoints_LuciferBot/" "$DST/Checkpoints_LuciferBot/"
    cp -f "$SRC/training_log.csv" "$DST/" 2>/dev/null
    cp -f "$SRC/curriculum_state.json" "$DST/" 2>/dev/null
else
    # Full sync: scripts + checkpoints + logs (exclude heavy/temp files)
    rsync -a --delete \
        --exclude='venv/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        --exclude='*.egg-info/' \
        --exclude='build/' \
        --exclude='runs/' \
        "$SRC/" "$DST/"
fi
