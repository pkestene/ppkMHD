#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Load vtk image data file (.vti). 
Compute error with reference file.
"""

import sys
import os

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import matplotlib.pyplot as plt

# test if 2 VTI files were given on the command line
if len(sys.argv)<3:
    sys.exit('Usage: %s file1.vti file2.vti' % sys.argv[0])

# check that files exist
if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: file %s was not found!' % sys.argv[1])
if not os.path.exists(sys.argv[2]):
    sys.exit('ERROR: file %s was not found!' % sys.argv[2])

f1=sys.argv[1]
f2=sys.argv[2]
    
# open vti files
print 'Reading data {} {}'.format(f1, f2)
reader1 = vtk.vtkXMLImageDataReader()
reader1.SetFileName(f1)
reader1.Update()

reader2 = vtk.vtkXMLImageDataReader()
reader2.SetFileName(f2)
reader2.Update()

# retrieve data dimensions
im1 = reader1.GetOutput()
rows1, cols1, _ = im1.GetDimensions()
im2 = reader2.GetOutput()
rows2, cols2, _ = im2.GetDimensions()

nx=rows1-1
ny=cols1-1

# retrieve density 'rho' and reshaped
rho1 = vtk_to_numpy ( im1.GetCellData().GetArray(0) )
rho1 = rho1.reshape(rows1-1, cols1-1)
rho2 = vtk_to_numpy ( im2.GetCellData().GetArray(0) )
rho2 = rho2.reshape(rows2-1, cols2-1)

# display
rho_diff=rho1-rho2
l1=np.sum(np.abs(rho_diff))/nx/ny
l2=np.linalg.norm(rho_diff)/nx/ny
#l22=np.sqrt(np.sum(np.abs(rho_diff)**2))/nx/ny

#print("L1 / L2 error {} {} {}".format(l1,l2,l22))
print("L1 / L2 error {} {}".format(l1,l2))

#plt.imshow(rho_diff)
#cbar = plt.colorbar()
#plt.show()
