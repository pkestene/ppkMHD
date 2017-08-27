#!/usr/bin/env python

"""
Just cross-check some Euler formulas (flux eigenmatrix)
"""

# Standard library imports
import sys
import numpy as np

import random as rd

rd.seed(12)

def test2d():

    rho = rd.random()
    c =  rd.random()
    v = rd.random()

    print("rho={} c={} v={}".format(rho,c,v))
    
    # left eigen matrix of x flux
    La = np.array(
        [[0.0, -rho/(2*c), 0.0,   1.0/(2*c**2)],
         [1.0/(1-rho*v),   0.0,  -rho/(1-rho*v),   -1.0/(1-rho*v)/c**2],
         [-v/(1-rho*v),    0.0,   1/(1-rho*v),        v/(1-rho*v)/c**2],
         [0.0, rho/(2*c),  0.0,   1.0/(2*c**2)]])
    print(La)

    print()
    
    # right eigen matrix of x flux
    Ra = np.array(
      [[1.0,    1.0, rho,   1.0],
       [-c/rho, 0.0, 0.0,   c/rho],
       [0.0,    v,   1.0, 0.0],
       [c**2,   0.0, 0.0,   c**2]])
    print(Ra)

    print("La*Ra:\n")
    print(np.dot(La,Ra))

    print("Ra*La:\n")
    print(np.dot(Ra,La))

    print(np.linalg.inv(Ra))

    
if __name__ == "__main__":

    print("Test 2d")
    test2d()
