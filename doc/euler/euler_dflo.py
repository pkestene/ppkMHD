#!/usr/bin/env python

"""
Just cross-check some Euler formulas (flux eigenmatrix)
"""

# Standard library imports
import sys
import numpy as np

import random as rd

def test_dflo():

    rd.seed(12)
    rho = rd.random()+1.0
    E = rd.random()
    c = rd.random()
    u = rd.random()
    v = rd.random()
    w = rd.random()
    g = rd.random()+1.0

    g1   = g - 1.0
    q2   = u*u + v*v
    p    = g1 * (E - 0.5 * rho * q2)
    c2   = g * p / rho
    c    = np.sqrt(c2)
    beta = 0.5/c2
    phi2 = 0.5*g1*q2
    h    = c2/g1 + 0.5*q2

    Rx=np.zeros((4,4))
    Lx=np.zeros((4,4))
    Ry=np.zeros((4,4))
    Ly=np.zeros((4,4))

    Rx[0][0] = 1;      Rx[0][1] = 0;  Rx[0][2] = 1;     Rx[0][3] = 1;
    Rx[1][0] = u;      Rx[1][1] = 0;  Rx[1][2] = u+c;   Rx[1][3] = u-c;
    Rx[2][0] = v;      Rx[2][1] = -1; Rx[2][2] = v;     Rx[2][3] = v;
    Rx[3][0] = 0.5*q2; Rx[3][1] = -v; Rx[3][2] = h+c*u; Rx[3][3] = h-c*u;

    Ry[0][0] = 1;      Ry[0][1] = 0;  Ry[0][2] = 1;     Ry[0][3] = 1;
    Ry[1][0] = u;      Ry[1][1] = 1;  Ry[1][2] = u;     Ry[1][3] = u;
    Ry[2][0] = v;      Ry[2][1] = 0;  Ry[2][2] = v+c;   Ry[2][3] = v-c;
    Ry[3][0] = 0.5*q2; Ry[3][1] = u;  Ry[3][2] = h+c*v; Ry[3][3] = h-c*v;

    Lx[0][0] = 1-phi2/c2;       Lx[0][1] = g1*u/c2;       Lx[0][2] = g1*v/c2;    Lx[0][3] = -g1/c2;
    Lx[1][0] = v;               Lx[1][1] = 0;             Lx[1][2] = -1;         Lx[1][3] = 0;
    Lx[2][0] = beta*(phi2-c*u); Lx[2][1] = beta*(c-g1*u); Lx[2][2] = -beta*g1*v; Lx[2][3] = beta*g1;
    Lx[3][0] = beta*(phi2+c*u); Lx[3][1] =-beta*(c+g1*u); Lx[3][2] = -beta*g1*v; Lx[3][3] = beta*g1;

    Ly[0][0] = 1-phi2/c2;       Ly[0][1] = g1*u/c2;       Ly[0][2] = g1*v/c2;       Ly[0][3] = -g1/c2;
    Ly[1][0] = -u;              Ly[1][1] = 1;             Ly[1][2] = 0;             Ly[1][3] = 0;
    Ly[2][0] = beta*(phi2-c*v); Ly[2][1] =-beta*g1*u;     Ly[2][2] = beta*(c-g1*v); Ly[2][3] = beta*g1;
    Ly[3][0] = beta*(phi2+c*v); Ly[3][1] =-beta*g1*u;     Ly[3][2] =-beta*(c+g1*v); Ly[3][3] = beta*g1;

    print("1-phi2/c2={}".format(1-phi2/c2))

    print("Checking that Lx.Rx=identy")
    print(np.dot(Lx,Rx))



if __name__ == "__main__":

    test_dflo()
