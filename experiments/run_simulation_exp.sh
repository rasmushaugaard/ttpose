#!/usr/bin/env bash
set -e

for sample_err in "1e-5" "3e-5" "1e-4" "3e-4" "1e-3" 
do
    python -m ttpose.run \
        data/meshes/matlab_logo.ply \
        --sampling "uni:1000 > fps:10" \
        --sample-err-max $sample_err \
        --max-pose-bins 1e7 \
        --save-path data/logo_err_$sample_err.json \
        --seed 1
done