# Fixture calibration with guaranteed bounds from a few correspondence-free surface points

ICRA 2024, [paper on arxiv](https://arxiv.org/abs/2402.18123).

This repo includes an interface for Universal Robots using [ur_rtde](https://gitlab.com/sdurobotics/ur_rtde).  
It should, however, be relatively easy to adapt to other robot interfaces. See [robot.py](src/ttpose/robot.py).

## Demo

TODO: link to colab

## Install

```bash
$ pip install git+[this repo (https)]
```

If you're experiencing problems with the installation, try making a clean virtual environment first, e.g. with conda ([miniconda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)):
```bash
$ conda create -n ttpose python=3.10
$ conda activate ttpose
$ pip install git+[this repo (https)]
```

## Usage

With simulated surface samples:
```bash
$ python -m ttpose.run [path to mesh] --vis-so3 --vis-samples
```
See `ttpose/run.py` to see how to use the python interface.


## Replicating results in paper

Figure 3:
```bash
$ python -m ttpose.run cube --max-pose-bins 1e7 --vis-so3 --vis-samples
```
Replace `cube` by `cone` and `cyl`.


For the remaining results, you need the repo:

```bash
git clone [this repo]
cd ttpose/experiments
```

Figure 2:
```bash
$ ./run_simulation_exp.sh
$ ./analyze_sim_exp.py  # see ´logo_err.pdf´
```
Note that refactoring changed the order of calls to the random generator, so the figure is slightly different from the one in the paper.

Section IV.B.3:
```bash
$ ./run_real_exp.sh
$ ./analyze_real_exp.py
```

## Tool Tip

A FreeCAD drawing of the tool used in the paper experiments is located here: `experiments/tool.FCStd`.
The tip of the tool has an indent where a 3 mm (diameter) steel ball can be glued into place.
(Note that air and glue is supposed to be able to pass through the small drain in the tool tip to avoid building pressure under the steel ball).

It should be relatively easy to fixate steel balls into custom tools in a similar fashion. 


### Tool Tip Calibration

A calibration of the tool tip in the flange frame is required.
We calibrate the tip on a flat table. We use an industrial table with holes, so you'll see that `table_collect` collects points in a pattern avoiding those holes.
Follow the guides for the following scripts (in this order):

```bash
$ python -m ttpose.calib.tip_manual --help
$ python -m ttpose.calib.table_manual --help
$ python -m ttpose.calib.table --help
$ python -m ttpose.calib.tip --help
```


## Computing the minkowski sum

The minkowski sum of a fixture and a steel ball can be computed with the following scad script.

```scad
// minkowski.scad
minkowski(){
    sphere(d=[ball diameter], $fn=15);
    import("[path to fixture model]");
}
```

```bash
openscad minkowski.scad -o fixture_ball_minkowski.stl
```

The computation can be expensive, and it may be helpful to simplify the fixture mesh before running the minkowski sum.

If anyone is aware of a more efficient implementation of the minkowski sum, please let me know.

If computing the minkowski sum is a dealbreaker, note that you alternatively could estimate the normal at ball-fixture contact to figure out where on the steel ball the contact is.
This would then allow sampling points directly on the fixture surface instead of on the minkowski sum, but with additional uncertainties from normal estimation.


## Citation

```bibtex
@inproceedings{haugaard2024fixture,
  title={Fixture calibration with guaranteed bounds from a few correspondence-free surface points},
  author={Haugaard, Rasmus Laurvig and Kim, Yitaek and Iversen, Thorbj{\o}rn Mosekj{\ae}r},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2024},
  organization={IEEE}
}
```