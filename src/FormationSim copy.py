#!/usr/bin/env python3
from random import random
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from scipy import integrate
import scipy
import matplotlib.pyplot as plt
import csv
from QuadSys import QuadSys

def mysys(t, X):
    x = X[0]
    
    for i in range(xn-1):
        x = np.vstack((x, X[i+1]))

    for idx in range(N_agent):        
        u[idx*2] = u[idx*2] + random()/1000000
        u[idx*2+1] = u[idx*2+1] + random()/1000000
        
    dx = np.matmul(A0, x) + np.matmul(B0, u)

    dxx = np.kron(x, x)
    dux = np.kron(x, u)
    dX = np.array(dx)
    dX = np.vstack((dX, dxx))
    dX = np.vstack((dX, dux))

    return np.transpose(dX)[0]

def fill_diagonal(N_agent: int, D:np.array):
    row = len(D)
    col = len(D[0])
    An = np.zeros((row*N_agent, col*N_agent))
    for idx in range(N_agent):
        for j in range(row):
            for k in range(col):
                An[idx*row+j][idx*col+k] = D[j][k]
    return An

##############################################################################################
#### User Define ####

save_file = 0
N_agent = 3

hover = 0
l = 1                        # length of explore window, should be greater than xn^2, timestep dT
fly_time = 10                 # length of fly window, timestep dT
iter_max = 50                   # max iteration to compute
dT = 0.1                       # timestep

epsilon = 10e-3                 # stop threshold

configDict = {"dt": 0.1, "stepNumHorizon": 10, "startPointMethod": "zeroInput"}
buildFlag = True
MyQuad = QuadSys(configDict, buildFlag)

# x0 = np.array([[-1.5],[-0.5],[0],[0],
#                [-1.5],[-1.5],[0],[0],
#                [-1.5],[0.5],[0],[0]])

x0 = np.array([[0.5],[0.0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [-0.5],[-0.5],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
               [-0.5],[0.5],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

target = np.array([[0],[0]])  # target
distance = np.array([[-0.707],[-0.5],[-0.707],[0.5]])  # formation


g = 9.81
m = 0.2
Ixx = 0.1
Iyy = 0.1
Izz = 0.1

A = np.array([[0,0,0,1,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,-g,0,0,0,0],
             [0,0,0,0,0,0,g,0,0,0,0,0],
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


n = len(A)*N_agent
xn = len(B)*N_agent                     # number of rows
un = len(B[0])*N_agent                  # number of columns
single_xn = int(xn/N_agent)

##### desire state #######
desire = np.array([[-0.707],[-0.5],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
                   [-0.707],[0.5],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],
                   [1.5],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])

# desire = np.array([1],[2],[0],[0])
# you may just define your desire state directly instead of using this loop
#################

############ Weight matrix #################
Q = np.eye(xn)*1
Q[(N_agent-1)*4][(N_agent-1)*4] = 0.1
Q[(N_agent-1)*4+1][(N_agent-1)*4+1] = 0.1
R = np.eye(un)*1
##### just give as Q = something instead of using this loop
############################################

An = fill_diagonal(N_agent, A)
Bn = fill_diagonal(N_agent, B)

A = An
B = Bn

T = np.zeros((xn, xn))
for idx in range(N_agent-1):
    for k in range(single_xn):
        T[idx*single_xn+k][k] = -1
        T[idx*single_xn+k][(idx+1)*single_xn+k] = 1
for idx in range(single_xn):
    for k in range(N_agent):
        T[(N_agent-1)*single_xn+idx][k*single_xn+idx] = 1/N_agent

A0 = np.matmul(np.matmul(T, A), inv(T))
B0 = np.matmul(T, B)

u = np.zeros((un, 1))

X_save = np.transpose(x0)
x0 = np.matmul(T,x0)
X = x0
X = np.vstack((X, np.transpose(np.kron(np.transpose(x0), np.transpose(x0)))))
X = np.vstack((X, np.kron(x0, np.zeros((un, 1)))))

t_save = np.zeros(1)
v_save = np.zeros((2*N_agent,1))

for idx in range(hover):
    X_save = np.vstack((X_save,X_save[-1]))
    t_save = np.hstack((t_save, t_save[-1]+dT))

X = np.transpose(X)[-1]

X0 = np.matrix(scipy.linalg.solve_continuous_are(A0, B0, Q, R))
K0 = np.matrix(scipy.linalg.inv(R)*(B0.T*X0))

# print("Optimal K")
# print(K0)

#### Formation ##########
x = np.reshape(X_save[-1], (xn,1))
for idx in range(fly_time):
    z = np.matmul(T, x)
    # if idx > 1000:
    #     desire[24] = desire[24]+0.01
    #     desire[25] = desire[25]-0.02
    X = z - desire
    # u = -np.matmul(K, X)
    u = -np.matmul(K0, X)
    dx = np.matmul(A, x) + np.matmul(B, u)

    # print(dx)
    new_v = np.zeros((2*N_agent,1))
    for idx in range(N_agent):
        new_v[idx*2] = dx[idx*4]
        new_v[idx*2+1] = dx[idx*4+1]

    v_save = np.hstack((v_save, new_v))

    x = x + dx*dT
    # new_x = MyQuad._discDynFun(x[0:12], u[0:4])
    
    # for kdx in range(N_agent-1):
    #     new_x = np.vstack((new_x, MyQuad._discDynFun(x[(idx+1)*12:(idx+1)*12+12], u[(idx+1)*4:(idx+1)*4+4])))
    # x = new_x

    t_save = np.hstack((t_save, t_save[-1]+dT))
    X_save = np.vstack((X_save, np.transpose(x)))

fig, ax = plt.subplots()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Formation')
fig_legend = ['Target']

ax.plot(target[0], target[1], "*")
ax.legend(['Target'])
for idx in range(N_agent):
    ax.plot(X_save[:,idx*single_xn], X_save[:,idx*single_xn+1])
    fig_legend.append('Agent')
    
ax.legend(fig_legend)
# ax.set(xlim=(-6, 6), ylim=(-6, 6))

plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel('iter')
# ax.set_title('|K-K*|')
# ax.plot(iter_save, K_save, marker='.')
# plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel('iter')
# ax.set_title('|P-P_old|')
# ax.plot(iter_save, P_save, marker='.')
# plt.show()

# with open('trajectory.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(t_save)
#     for idx in range(N_agent):
#         writer.writerow(X_save[:,idx*4])
#         writer.writerow(X_save[:,idx*4+1])

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
            new_v[j*2] = (x_fly[idx+1][j*4+2]-x_fly[idx][j*4+2])/(dT*10)
            new_v[j*2+1] = (x_fly[idx+1][j*4+3]-x_fly[idx][j*4+3])/(dT*10) 
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

