#!/bin/bash
name="2023_03_11_13_00_00"
n_trials="2000"
for k in "1" "2" "4"; do
    for seed in "0" "1" "2"; do
        for category in "carpet" "grid" "leather" "tile" "wood" "bottle" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper"; do
            for test_set_search in "False" "True"; do
                sbatch search.sh "${name}_n${n_trials}_k${k}_s${seed}_${category}_${test_set_search}" "${n_trials}" "${k}" "${seed}" "${category}" "${test_set_search}" "" "" ""
            done
        done
    done
done
