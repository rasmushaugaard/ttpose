#!/usr/bin/env bash
set -e

for name in "student_1" "student_2" "student_3" "expert"
do
    python -m ttpose.save_sample_pts data/samples/$name
    python -m ttpose.run \
        data/meshes/matlab_logo_minkowski.ply \
        --samples data/samples/$name.npy \
        --sample-err-max 1e-3 \
        --max-pose-bins 1e7 \
        --save-path data/$name.json
done