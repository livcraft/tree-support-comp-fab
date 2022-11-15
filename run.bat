@echo off
rem Brute-force voxelization
python main.py spot bf 0.125

rem Accelerated voxelization
python main.py spot fast 0.125

rem Marching cubes
python main.py spot mc