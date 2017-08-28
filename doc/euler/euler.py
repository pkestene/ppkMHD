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


def test3d_A():

    rho = rd.random()
    c =  rd.random()
    u = rd.random()
    v = rd.random()
    w = rd.random()

    print("rho={} c={} u={} v={} w={}".format(rho,c,u,v,w))
    
    # right eigen matrix of x flux
    Ra = np.array(
      [[1.0,    1.0, rho,   rho,  1.0],
       [-c/rho, 0.0, 0.0,   0.0,  c/rho],
       [0.0,    v,   1.0,   v,    0.0],
       [0.0,    w,   w,     1.0,  0.0],
       [c**2,   0.0, 0.0,   0.0,  c**2]])
    print("Ra:")
    print(Ra)

    D = rho*(v-1)*(w-1)+(1-rho)*(1-v*w)
    
    # left eigen matrix of x flux
    La = np.array(
        [[ 0.0,       -rho/(2*c),     0.0,                  0.0,                   1.0/(2*c**2)],
         [ (1-v*w)/D,   0.0,   -rho*(1-w)/D,   -rho*(1-v)/D, -(1-v*w)/D/c**2],
         [-(1-w)*v/D,   0.0,          (1-v)/D,   rho*(1.0-v)/D,                    v/D/c**2],
         [      -w/D,   0.0,       -rho/D,   rho/(1-rho*v-rho*w),    w/D/c**2],
         [0.0,         rho/(2*c),      0.0,                  0.0,                    1.0/(2*c**2)]])
    print("La:")
    print(La)

    print()
    print("1/(1-rho*v-rho*w)={}".format(1.0/(1.0-rho*v-rho*w)))
    print("-rho/2/c={}".format(-rho/2/c))
    print("1.0/(2*c**2)={}".format(1.0/(2*c**2)))
    print("w/(1-rho*v-rho*w)={}".format(w/(1.0-rho*v-rho*w)))
    print()
    
    print("La*Ra:\n")
    print(np.dot(La,Ra))

    # print("Ra*La:\n")
    # print(np.dot(Ra,La))

    print("inv(Ra)")
    print(np.linalg.inv(Ra))

    KK = np.array(
        [[ 1, rho, rho],
         [ v, 1, v],
         [w, w, 1]])

    print(np.linalg.inv(KK))
    
    alpha1=  (1-v*w)/D
    beta1  = -rho*(1-w)/D
    gamma1 = -rho*(1-v)/D
    print("alpha1={}".format(alpha1))
    print("beta1={}".format(beta1))
    print("gamma1={}".format(gamma1))
    
if __name__ == "__main__":

    #print("Test 2d")
    #test2d_B()
    print("Test 3d")
    test3d_A()
