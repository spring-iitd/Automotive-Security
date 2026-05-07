#!/bin/bash
set -e

TOTAL_ROUNDS=4

for ((i=0; i<$TOTAL_ROUNDS; i++))
do
    echo "======================================="
    echo "Running Round $i"
    echo "======================================="
    python driver_spoof_CH.py --round $i
done


# python driver_spoof_CH.py --round 0