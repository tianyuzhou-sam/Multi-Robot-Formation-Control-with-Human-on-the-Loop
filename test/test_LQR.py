#!/usr/bin/env python3
from random import random
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import integrate
import scipy
import control
import matplotlib.pyplot as plt
import csv
import os
import sys
sys.path.append(os.getcwd()+'/src')
from QuadSys import QuadSys

##############################################################################################
#### User Define ####
save_file = 0

fly_time = 1000                # length of fly window, timestep dt
dt = 0.1                       # timestep

configDict = {"dt": dt, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}
g = 9.81
m = 0.2
Ixx = 8.1 * 1e-3
Iyy = 8.1 * 1e-3
Izz = 14.2 * 1e-3

MyQuad = QuadSys(configDict, g, m, Ixx, Iyy, Izz)

x0 = np.array([[0.],[0.0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])



A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
              [0,0,0,0,1,0,0,0,0,0,0,0],
              [0,0,0,0,0,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,g,0,0,0,0],
              [0,0,0,0,0,0,-g,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0,0,0]])

B = np.array([[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [1/m,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,0,0,0],
             [0,1/Ixx,0,0],
             [0,0,1/Iyy,0],
             [0,0,0,1/Izz]])

##############################################################################################

xn = len(B)                     # number of rows
un = len(B[0])                  # number of columns

A = np.eye(xn) + dt*A
B = dt*B

##### desire state #######
desire = np.array([[1],[-1.],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
#################

############ Weight matrix #################
Q = np.eye(xn)*10
R = np.eye(un)*1
############################################

u = np.zeros((un, 1))

X_save = np.transpose(x0)

X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))

t_save = np.zeros(1)
v_save = np.zeros((6,1))

X = np.transpose(X)[-1]

P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
K = np.matrix(np.matmul(scipy.linalg.inv(R + B.T*P*B),B.T)*P*A)
print(K)

x = np.reshape(X_save[-1], (xn,1))
for idx in range(fly_time):

    z = x - desire
    u = -np.matmul(K, z)
    dx = np.matmul(A, x) + np.matmul(B, u)

    new_v = np.vstack((dx[3:6], dx[9:12]))

    v_save = np.hstack((v_save, new_v))

    u[0] = u[0] + m*g

    x = MyQuad._discDynFun(x, u)

    t_save = np.hstack((t_save, t_save[-1]+dt))
    X_save = np.vstack((X_save, np.transpose(x)))


fig1, ax1 = plt.subplots(3,1)
ax1[0].plot(t_save, X_save[:,0])
ax1[0].set_xlabel('t (s)')
ax1[0].set_ylabel('x (m)')
ax1[1].plot(t_save, X_save[:,1])
ax1[1].set_xlabel('t (s)')
ax1[1].set_ylabel('y (m)')
ax1[2].plot(t_save, X_save[:,2])
ax1[2].set_xlabel('t (s)')
ax1[2].set_ylabel('z (m)')

fig2, ax2 = plt.subplots(3,1)
ax2[0].plot(t_save, X_save[:,7])
ax2[0].set_xlabel('t (s)')
ax2[0].set_ylabel('roll (rad))')
ax2[1].plot(t_save, X_save[:,8])
ax2[1].set_xlabel('t (s)')
ax2[1].set_ylabel('pitch (rad)')
ax2[2].plot(t_save, X_save[:,9])
ax2[2].set_xlabel('t (s)')
ax2[2].set_ylabel('yaw (rad)')


plt.show()

if save_file == 1:
    t_fly = t_save[::10]
    x_fly = X_save[::10]
    height = np.ones((len(x_fly)))*0.8
    v_z = np.zeros((len(x_fly)))
    v_save = v_save.T
    v_save = v_save[::10]

    v_save = np.zeros((2*N_agent,1))
    for idx in range(len(x_fly)-1):
        new_v = np.zeros((2*N_agent,1))
        for j in range(N_agent):
            new_v[j*2] = (x_fly[idx+1][j*4+2]-x_fly[idx][j*4+2])/(dt*10)
            new_v[j*2+1] = (x_fly[idx+1][j*4+3]-x_fly[idx][j*4+3])/(dt*10) 
        v_save = np.hstack((v_save,new_v))

    with open('trajectory1.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(t_fly)
        writer.writerow(np.asarray(x_fly[:,0]).flatten())
        writer.writerow(np.asarray(x_fly[:,1]).flatten())
        writer.writerow(height)
        writer.writerow(v_save[0])
        writer.writerow(v_save[1])
        writer.writerow(v_z)

    with open('trajectory2.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(t_fly)
        writer.writerow(np.asarray(x_fly[:,4]).flatten())
        writer.writerow(np.asarray(x_fly[:,5]).flatten())
        writer.writerow(height)
        writer.writerow(v_save[2])
        writer.writerow(v_save[3])
        writer.writerow(v_z)

    if N_agent == 3:
        with open('trajectory3.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(t_fly)
            writer.writerow(np.asarray(x_fly[:,8]).flatten())
            writer.writerow(np.asarray(x_fly[:,9]).flatten())
            writer.writerow(height)
            writer.writerow(v_save[4])
            writer.writerow(v_save[5])
            writer.writerow(v_z)

