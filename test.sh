#!/bin/bash
PY_SCRIPT="simulation/simu_online.py"
N_TEST=(600)
SET_IDS=(1)
Q_VALUES=(1)
REPEAT_TIME=100


for nt_id in "${N_TEST[@]}"; do
    for set_id in "${SET_IDS[@]}"; do
        for q in "${Q_VALUES[@]}"; do
             python "$PY_SCRIPT" "$nt_id" "$set_id" "$q" "$REPEAT_TIME"
        done
    done
done
