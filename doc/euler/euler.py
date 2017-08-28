#!/usr/bin/env python

"""
Just cross-check some Euler formulas (flux eigenmatrix)
"""

# Standard library imports
import sys
import numpy as np

import random as rd

rd.seed(12)

def test2d_A():

    rho = rd.random()
    c =  rd.random()
    u = rd.random()
    v = rd.random()

    print("rho={} c={} u={} v={}".format(rho,c,u,v))
    
    # left eigen matrix of x flux
    La = np.array(
        [[0.0,            -rho/(2*c),            0.0,   1.0/(2*c**2)],
         [1.0/(1-rho*v),   0.0,       -rho/(1-rho*v),  -1.0/(1-rho*v)/c**2],
         [-v/(1-rho*v),    0.0,          1/(1-rho*v),     v/(1-rho*v)/c**2],
         [0.0,             rho/(2*c),            0.0,   1.0/(2*c**2)]])
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

def test2d_B():

    rho = rd.random()
    c =  rd.random()
    u = rd.random()
    v = rd.random()

    print("rho={} c={} u={} v={}".format(rho,c,u,v))
    
    # left eigen matrix of x flux
    Lb = np.array(
        [[0.0,                        0.0,   -rho/(2*c),   1.0/(2*c**2)],
         [1.0/(1-rho*u),   -rho/(1-rho*u),          0.0, -1.0/(1-rho*u)/c**2],
         [ -u/(1-rho*u),      1/(1-rho*u),          0.0,    u/(1-rho*u)/c**2],
         [0.0,                        0.0,    rho/(2*c),   1.0/(2*c**2)]])
    print("\nLb:")
    print(Lb)

    print()
    print("1/(1-rho*u)={}".format(1.0/(1.0-rho*u)))
    print("rho/(1-rho*u)={}".format(rho/(1.0-rho*u)))
    print("u/(1-rho*u)={}".format(u/(1.0-rho*u)))
    print("rho*u/(1-rho*u)={}".format(rho*u/(1.0-rho*u)))
    print("u/(u-rho)={}".format(u/(u-rho)))
    
    # right eigen matrix of x flux
    Rb = np.array(
      [[1.0,    1.0, rho,   1.0],
       [0.0,      u, 1.0,   0.0],
       [-c/rho, 0.0, 0.0,   c/rho],
       [c**2,   0.0, 0.0,   c**2]])
    print("\nRb:")
    print(Rb)

    print("\nLb*Rb:")
    print(np.dot(Lb,Rb))

    print("\nRb*Lb:")
    print(np.dot(Rb,Lb))

    print("\nsum(Rb**(-1)-Lb)={}".format( np.sum(np.linalg.inv(Rb)-Lb)) )

    
if __name__ == "__main__":

    print("Test 2d")
    test2d_B()
