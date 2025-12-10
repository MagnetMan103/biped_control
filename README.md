# Quickstart

To run, first clone the repo
```
git clone https://github.com/MagnetMan103/biped_control
```
Then, install mujoco
```
pip install mujoco
```
To run the balance script, do:
```
mjpython balance.py
```
This script will have the robot balance around the origin.

To run the position control script, do:
```
mjpython position_control.py
```
This script will let you select a target position that the robot will move toward.
