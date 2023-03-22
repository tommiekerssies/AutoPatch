#!/bin/bash
name="ablation_2023_03_20_10_33_00"
n_trials="2000"
for k in "4"; do
    for seed in "0" "1" "2"; do
        for category in "carpet" "grid" "leather" "tile" "wood" "bottle" "cable" "capsule" "hazelnut" "metal_nut" "pill" "screw" "toothbrush" "transistor" "zipper"; do
            for test_set_search in "True"; do
                sbatch search.sh "${name}_n${n_trials}_k${k}_s${seed}_${category}_${test_set_search}" "${n_trials}" "${k}" "${seed}" "${category}" "${test_set_search}" "ofa_mbv3_d234_e346_k357_w1.2" "7" "6"
            done
        done
    done
done
