import numpy as np
from vispy import app
import pycsg as csg

app.use_app('pyqt5') # Set backend

from matplotlib import pyplot as plt
from pyslm.core import Part
import pyslm.support

import trimesh
import trimesh.creation
import logging

OVERHANG_ANGLE = 55 # deg - Overhang angle

img = np.zeros([1000,1000])
x,y = np.meshgrid(np.arange(0,img.shape[1]), np.arange(0, img.shape[0]))

solid = np.sqrt((x-500)**2 + (y-500)**2) < 400
bound =   np.sqrt((x-500)**2 + (y-500)**2)
#solid = img[:,:,3].astype(np.float64)
orient = np.array([1.0,1.0])
orient = orient / np.linalg.norm(orient)
perp = np.array((orient[1], orient[0]))


dotProd = np.dot(x, orient[0])+np.dot(y,orient[1])
solid2 = solid*( np.sin(0.2*dotProd))
bound * solid2
