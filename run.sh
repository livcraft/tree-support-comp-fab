#!/bin/bash
# Brute-force voxelization
python main.py spot bf 0.125

# Accelerated voxelization
python main.py spot fast 0.125

python main.py spot mc
