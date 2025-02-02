#!/usr/bin/env python

"""
Just cross-check some Euler formulas (flux eigenmatrix)
"""

# Standard library imports
import sys
import numpy as np

import random as rd

rd.seed(12)
rho = rd.random()
c =  rd.random()
u = rd.random()
v = rd.random()
w = rd.random()
g = rd.random()

def eigen_cons_1d():

    # enthalpy
    H = u*u/2+c*c/(g-1)

    # intermediate value: scaling factor for the left eigen vector matrix
    D = 2*c*(H-u*u/2)

    A = np.array([[0, 1, 0],
                  [0.5*(g-3)*u*u,(3-g)*u,g-1],
                  [u*(0.5*(g-1)*u*u-H),H-(g-1)*u*u, g*u]])

    print("Euler Jacobian matrix in 1d")
    print(A)

    print("Eigenvalues of A")
    print(np.linalg.eigvals(A))
    print("to be compared with u+c={} u={} and u-c={}".format(u+c,u,u-c))

    #printf("Determinant of A={} compared to ".format(np.linalg.det(A)))

    # eigenvalues matrix
    L_eig=np.array([[u-c,0,0],
                    [0,u,0],
                    [0,0,u+c]])


    # right eigenvectors
    R = np.array([[1,1,1],
                  [u-c,u,u+c],
                  [H-u*c,u*u/2,H+u*c]])


    # left eigenvectors (R^-1)
    L = np.array([[u*(H-u*u/2+c*u/2), -H-c*u+u*u/2,    c],
                  [2*c*(H-u*u),        2*c*u,       -2*c],
                  [u*(-H+c*u/2+u*u/2), H-c*u-u*u/2,    c]])/D

    # check
    print("Checking that L.R=identy")
    print(np.dot(L,R))

    print("Checking that Lamba = L.A.R")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(L,A),R)-L_eig)))

def eigen_cons_2d():

    # enthalpy
    H = (u*u+v*v)/2+c*c/(g-1)
    V2=u*u+v*v

    c2 = c*c
    g1 = g-1
    beta = 1.0/2/c2

    phi2 = g1*H-c2

    A = np.array([[0, 1, 0, 0],
                  [phi2-u*u, (3-g)*u, -g1*v, g1],
                  [-u*v, v,u,0],
                  [-u*(H-phi2),H-g1*u*u, -g1*u*v, g*u]])

    B = np.array([[0, 0, 1, 0],
                  [-u*v, v,u,0],
                  [phi2-v*v, -g1*u, (3-g)*v, g1],
                  [-v*(H-phi2),-g1*u*v, H-g1*v*v, g*v]])

    print("Euler Jacobian matrix in 2d : A")
    print(A)

    print("Eigenvalues of A")
    print(np.linalg.eigvals(A))
    print("to be compared with u+c={} u={} and u-c={}".format(u+c,u,u-c))

    #printf("Determinant of A={} compared to ".format(np.linalg.det(A)))

    # eigenvalues matrix
    eigA = np.array([[u-c,0,0,0],
                     [0,u,0,0],
                     [0,0,u,0],
                     [0,0,0,u+c]])

    eigB = np.array([[v-c,0,0,0],
                     [0,v,0,0],
                     [0,0,v,0],
                     [0,0,0,v+c]])


    # right eigenvectors
    Ra = np.array([[1,    1,    0, 1],
                   [u-c,  u,    0, u+c],
                   [v,    v,    1, v],
                   [H-u*c,V2/2, v, H+u*c]])

    Rb = np.array([[1,    1,    0, 1],
                   [u,    u,    1, u],
                   [v-c,  v,    0, v+c],
                   [H-v*c,V2/2, u, H+v*c]])

    # left eigenvectors (R^-1)
    La = np.array([[beta*(phi2+u*c), -beta*(g1*u+c), -beta*g1*v, beta*g1],
                   [1.0-phi2/c2, g1*u/c2, g1*v/c2, -g1/c2],
                   [-v,0,1,0],
                   [beta*(phi2-u*c), -beta*(g1*u-c), -beta*g1*v, beta*g1]])

    Lb = np.array([[beta*(phi2+v*c), -beta*g1*u, -beta*(g1*v+c), beta*g1],
                   [1.0-phi2/c2, g1*u/c2, g1*v/c2, -g1/c2],
                   [-u,1,0,0],
                   [beta*(phi2-v*c), -beta*g1*u, -beta*(g1*v-c), beta*g1]])

    # check
    print("Checking that La.Ra=identy")
    print(np.dot(La,Ra))
    print("norm of La.Ra-Id = {}".format(np.linalg.norm(np.dot(La,Ra)-np.eye(4))))

    print("Checking that Lb.Rb=identy")
    print(np.dot(Lb,Rb))
    print("norm of Lb.Rb-Id = {}".format(np.linalg.norm(np.dot(Lb,Rb)-np.eye(4))))

    print("Checking that eigA = La.A.Ra")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(La,A),Ra)-eigA)))

    print("Checking that eigB = Lb.B.Rb")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(Lb,B),Rb)-eigB)))

def eigen_cons_3d():

    # enthalpy
    H = (u*u+v*v+w*w)/2+c*c/(g-1)
    V2=u*u+v*v+w*w

    c2 = c*c
    g1 = g-1
    beta = 1.0/2/c2

    phi2 = g1*H-c2

    A = np.array([[0, 1, 0, 0, 0],
                  [phi2-u*u, (3-g)*u, -g1*v, -g1*w, g1],
                  [-u*v, v,u,0,0],
                  [-u*w, w,0,u,0],
                  [-u*(H-phi2),H-g1*u*u, -g1*u*v, -g1*u*w, g*u]])

    B = np.array([[0, 0, 1, 0, 0],
                  [-v*u, v,u,0,0],
                  [phi2-v*v, -g1*u, (3-g)*v, -g1*w, g1],
                  [-v*w, 0,w,v,0],
                  [-v*(H-phi2),-g1*v*u, H-g1*v*v, -g1*v*w, g*v]])

    C = np.array([[0, 0, 0, 1, 0],
                  [-w*u, w,0,u,0],
                  [-w*v, 0,w,v,0],
                  [phi2-w*w, -g1*u, -g1*v, (3-g)*w, g1],
                  [-w*(H-phi2),-g1*w*u, -g1*w*v, H-g1*w*w, g*w]])

    print("Euler Jacobian matrix in 2d : A")
    print(A)

    print("Eigenvalues of A")
    print(np.linalg.eigvals(A))
    print("to be compared with u+c={} u={} and u-c={}".format(u+c,u,u-c))

    print("Eigenvalues of B")
    print(np.linalg.eigvals(B))
    print("to be compared with v+c={} v={} and v-c={}".format(v+c,v,v-c))

    print("Eigenvalues of C")
    print(np.linalg.eigvals(C))
    print("to be compared with w+c={} w={} and w-c={}".format(w+c,w,w-c))

    # eigenvalues matrix
    eigA = np.array([[u-c,0,0,0,0],
                     [0,u,0,0,0],
                     [0,0,u,0,0],
                     [0,0,0,u,0],
                     [0,0,0,0,u+c]])

    eigB = np.array([[v-c,0,0,0,0],
                     [0,v,0,0,0],
                     [0,0,v,0,0],
                     [0,0,0,v,0],
                     [0,0,0,0,v+c]])

    eigC = np.array([[w-c,0,0,0,0],
                     [0,w,0,0,0],
                     [0,0,w,0,0],
                     [0,0,0,w,0],
                     [0,0,0,0,w+c]])



    # right eigenvectors
    Ra = np.array([[1,    1,    0, 0, 1],
                   [u-c,  u,    0, 0, u+c],
                   [v,    v,    1, 0, v],
                   [w,    w,    0, 1, w],
                   [H-u*c,V2/2, v, w, H+u*c]])

    Rb = np.array([[1,    1,    0, 0, 1],
                   [u,    u,    1, 0, u],
                   [v-c,  v,    0, 0, v+c],
                   [w,    w,    0, 1, w],
                   [H-v*c,V2/2, u, w, H+v*c]])

    Rc = np.array([[1,    1,    0, 0, 1],
                   [u,    u,    1, 0, u],
                   [v  ,  v,    0, 1, v],
                   [w-c,  w,    0, 0, w+c],
                   [H-w*c,V2/2, u, v, H+w*c]])

    # left eigenvectors (R^-1)
    La = np.array([[beta*(phi2+u*c), -beta*(g1*u+c), -beta*g1*v,    -beta*g1*w,     beta*g1],
                   [1.0-phi2/c2,     g1*u/c2,        g1*v/c2,       g1*w/c2,        -g1/c2],
                   [-v,              0,              1,             0,              0],
                   [-w,              0,              0,             1,              0],
                   [beta*(phi2-u*c), -beta*(g1*u-c), -beta*g1*v,    -beta*g1*w,     beta*g1]])

    Lb = np.array([[beta*(phi2+v*c), -beta*g1*u,    -beta*(g1*v+c), -beta*g1*w,     beta*g1],
                   [1.0-phi2/c2,     g1*u/c2,       g1*v/c2,        g1*w/c2,        -g1/c2],
                   [-u,              1,             0,              0,              0],
                   [-w,              0,             0,              1,              0],
                   [beta*(phi2-v*c), -beta*g1*u,    -beta*(g1*v-c), -beta*g1*w,     beta*g1]])

    Lc = np.array([[beta*(phi2+w*c), -beta*g1*u, -beta*g1*v,        -beta*(g1*w+c), beta*g1],
                   [1.0-phi2/c2,     g1*u/c2,    g1*v/c2,           g1*w/c2,        -g1/c2],
                   [-u,              1,          0,                 0,              0],
                   [-v,              0,          1,                 0,              0],
                   [beta*(phi2-w*c), -beta*g1*u, -beta*g1*v,        -beta*(g1*w-c), beta*g1]])

    # check
    print("Checking that La.Ra=identy")
    print(np.dot(La,Ra))
    print("norm of La.Ra-Id = {}".format(np.linalg.norm(np.dot(La,Ra)-np.eye(5))))

    print("Checking that Lb.Rb=identy")
    print(np.dot(Lb,Rb))
    print("norm of Lb.Rb-Id = {}".format(np.linalg.norm(np.dot(Lb,Rb)-np.eye(5))))

    print("Checking that Lc.Rc=identy")
    print(np.dot(Lc,Rc))
    print("norm of Lc.Rc-Id = {}".format(np.linalg.norm(np.dot(Lc,Rc)-np.eye(5))))

    print("Checking that eigA = La.A.Ra")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(La,A),Ra)-eigA)))

    print("Checking that eigB = Lb.B.Rb")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(Lb,B),Rb)-eigB)))

    print("Checking that eigB = Lc.C.Rc")
    print("norm of difference is {}".format(np.linalg.norm(np.dot(np.dot(Lc,C),Rc)-eigC)))

def test2d_A():


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
         [ (1-v*w)/D,   0.0,   -rho*(1-w)/D,   -rho*(1-v)/D,    -(1-v*w)/D/c**2],
         [-(1-w)*v/D,   0.0,    (1-rho*w)/D,   -v*(1.0-rho)/D,   (1-w)*v/D/c**2],
         [-(1-v)*w/D,   0.0,    -w*(1.0-rho)/D, (1-rho*v)/D,  (1-v)*w/D/c**2],
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
    print(np.sum(np.dot(La,Ra)))

    # print("Ra*La:\n")
    # print(np.dot(Ra,La))

    print("inv(Ra)")
    print(np.linalg.inv(Ra))

def test3d_B():


    print("rho={} c={} u={} v={} w={}".format(rho,c,u,v,w))

    # right eigen matrix of x flux
    Rb = np.array(
      [[1.0,    rho,  1.0,  rho,  1.0],
       [0.0,    1.0,    u,  u,    0.0],
       [-c/rho, 0.0, 0.0,   0.0,  c/rho],
       [0.0,    w,   w,     1.0,  0.0],
       [c**2,   0.0, 0.0,   0.0,  c**2]])
    print("Rb:")
    print(Rb)

    D = rho*(w-1)*(u-1)+(1-rho)*(1-w*u)

    # left eigen matrix of x flux

    Lb = np.array(
        [[0.0, 0.0, -rho/2/c, 0.0, 1.0/2/c**2],
         [(-(1-w)*u)/D, (1-rho*w)/D, 0.0, -u*(1-rho)/D, (1-w)*u/D/c**2],
         [(1-w*u)/D,  -(rho*(1-w))/D, 0.0, -(rho*(1-u))/D, -(1 -w*u)/D/c**2],
         [-(1-u)*w/D, -(w*(1-rho))/D, 0.0, (1-rho*u)/D,  (1 - u)*w/D/c**2],
         [0.0, 0.0, rho/2/c, 0.0, 1.0/2/c**2]])


    print("Lb:")
    print(Lb)

    print()
    print("1/(1-rho*v-rho*w)={}".format(1.0/(1.0-rho*v-rho*w)))
    print("-rho/2/c={}".format(-rho/2/c))
    print("1.0/(2*c**2)={}".format(1.0/(2*c**2)))
    print("w/(1-rho*v-rho*w)={}".format(w/(1.0-rho*v-rho*w)))
    print()

    print("Lb*Rb:\n")
    print(np.dot(Lb,Rb))
    print(np.sum(np.dot(Lb,Rb)))

    # print("Rb*Lb:\n")
    # print(np.dot(Rb,Lb))

    print("inv(Rb)")
    print(np.linalg.inv(Rb))

def test3d_C():

    print("rho={} c={} u={} v={} w={}".format(rho,c,u,v,w))

    # right eigen matrix of x flux
    Rc = np.array(
      [[1.0,    rho,  rho,  1.0,  1.0],
       [0.0,    1.0,  u,    u,    0.0],
       [0.0,      v,  1.0,  v,    0.0],
       [-c/rho, 0.0,  0.0,  0.0,  c/rho],
       [c**2,   0.0,  0.0,  0.0,  c**2]])
    print("Rc:")
    print(Rc)

    D = rho*(u-1)*(v-1)+(1-rho)*(1-u*v)

    # left eigen matrix of x flux

    Lc = np.array(
        [[0.0, 0.0,  0.0, -rho/2/c, 1.0/2/c**2],
         [-(1-v)*u/D, (1-rho*v)/D, -(u*(1-rho))/D, 0.0,  (1 - v)*u/D/c**2],
         [(-(1-u)*v)/D, -v*(1-rho)/D, (1-rho*u)/D, 0.0,  (1-u)*v/D/c**2],
         [(1-u*v)/D, -(rho*(1-v))/D, -(rho*(1-u))/D, 0.0, -(1 -u*v)/D/c**2],
         [0.0, 0.0, 0.0, rho/2/c,  1.0/2/c**2]])


    print("Lc:")
    print(Lc)

    print("Lc*Rc:\n")
    print(np.dot(Lc,Rc))
    print(np.sum(np.dot(Lc,Rc)))

    # print("Rc*Lc:\n")
    # print(np.dot(Rc,Lc))

    print("inv(Rc)")
    print(np.linalg.inv(Rc))

if __name__ == "__main__":

    #print("Test 2d")
    #test2d_B()
    #print("Test 3d")
    #test3d_C()

    #eigen_cons_1d()
    #eigen_cons_2d()
    eigen_cons_3d()
