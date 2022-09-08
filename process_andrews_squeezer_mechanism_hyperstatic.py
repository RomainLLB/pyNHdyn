# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:56:32 2019

@author: rlecuyer
"""

import benchmark # cython module
from numpy import cos,sin,arctan2
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{commath}\usepackage{mathtools}') 


wdir=os.path.abspath('')

taille= 20
plt.rc('font', size=taille)


#%%############################################################################
# Parameters of the simulation
###############################################################################

    
h=1e-5 # time step in second
duration=0.03 
t=np.arange(0,duration+h,h)

# Spring parameters
ka = 4530 
l0a = 0.07785 
# Motor torque
T0 = 0.033 

#%%############################################################################
# Mechanical characteristics
###############################################################################

O=np.array([0,0]) 
A=np.array([-0.06934,-0.00227]) 
A1=A.copy();A2=A.copy()
B=np.array([-0.03635,0.03273])
C=np.array([0.014,0.072])
D=np.array([-0.01047,0.02536])
E=np.array([-0.034,0.01646])
F=np.array([-0.03163,-0.01562])
G=np.array([0.00699,-0.00043])
P=np.array([-0.02096,0.0013])
P1=P.copy();P2=P.copy();P3=P.copy();P4=P.copy();P5=P.copy();P6=P.copy()


# Initial angles
theta1_0 = arctan2(G[-1]-O[-1],G[0]-O[0])
theta2_0 = arctan2(G[-1]-P[-1],G[0]-P[0])
PBn=(B-P)/np.linalg.norm(B-P) # ( -cos(theta),sin(theta) )
theta3_0 = arctan2(PBn[1],-PBn[0])
theta4_0 = arctan2(P[-1]-E[-1],P[0]-E[0])
theta5_0 = arctan2(E[-1]-A[-1],E[0]-A[0])
theta6_0 = arctan2(P[-1]-F[-1],P[0]-F[0])
theta7_0 = arctan2(F[-1]-A[-1],F[0]-A[0])


# Mass and inertias
m1=0.04325;I1=2.194e-6;x1_0=0.0009182;y1_0=0.000057					
m2=0.00365;I2=4.410e-7;x2_0=-0.004491;y2_0=0.0002788					
m3=0.02373;I3=5.255e-6;x3_0=-0.01874;y3_0=0.02048					
m4=0.00706;I4=5.667e-7;x4_0=-0.03022;y4_0=0.01207					
m5=0.07050;I5=1.169e-5;x5_0=-0.05324;y5_0=0.01663					
m6=0.00706;I6=5.667e-7;x6_0=-0.02854;y6_0=-0.01072					
m7=0.05498;I7=1.912e-5;x7_0=-0.05926;y7_0=-0.0106					


MA = np.diag([m1,m1,I1,
              m2,m2,I2,
              m3,m3,I3,
              m4,m4,I4,
              m5,m5,I5,
              m6,m6,I6,
              m7,m7,I7])


#%%############################################################################
# Hyperstatic system
###############################################################################


# Parameter dictionnary
parameters=dict()
with open("input_parameters.txt", 'r') as f:
    tab = f.readlines()
for i in range(len(tab)):
    exec(tab[i])
    
# get the values of n and m from parameters
n = parameters["n"]
m = parameters["m"]

# Create the state space vector at t=0s
X0=np.zeros([2*n+m])
X0[n:2*n]=np.array([x1_0,y1_0,theta1_0,x2_0,y2_0,theta2_0,x3_0,y3_0,theta3_0,
  x4_0,y4_0,theta4_0,x5_0,y5_0,theta5_0,x6_0,y6_0,theta6_0,x7_0,y7_0,theta7_0]) 
    
# mass matrix and identity matrix  
parameters["MA"] = MA
parameters["Inn"] = np.eye(n)

# Motor torque
parameters["torque"] = T0

# Spring parameters
parameters["ka"]=ka
parameters["l0a"]=l0a


# hyperstatic system
dXsol, Xsol, derive = benchmark.RK45(t, X0, parameters,solver='dgelsd') 

F_c = Xsol[2*n:,:]

drift = np.zeros([t.size])
for i in range(t.size):
    drift[i] = np.abs( np.dot(F_c[:,i],derive[:,i]) )




#%%############################################################################
# Plots
###############################################################################


# Motion of point P 
sol={}
l_solides = [1, 2, 3, 4, 5, 6, 7]
for i in l_solides:
    keyx='x{}'.format(i)
    keyy='y{}'.format(i)
    keytheta='theta{}'.format(i)
    sol[keyx]=Xsol[n+3*(i-1)+0,:]
    sol[keyy]=Xsol[n+3*(i-1)+1,:]
    sol[keytheta]=Xsol[n+3*(i-1)+2,:]
X2P=parameters["X2P1"];Y2P=parameters["Y2P1"]
X3P=parameters["X3P1"];Y3P=parameters["Y3P1"]
X4P=parameters["X4P2"];Y4P=parameters["Y4P2"]
X6P=parameters["X6P3"];Y6P=parameters["Y6P3"]
P0_res = np.zeros([Xsol.shape[1],2])
P1_res = np.zeros([Xsol.shape[1],2])
P2_res = np.zeros([Xsol.shape[1],2])
P3_res = np.zeros([Xsol.shape[1],2])
P0_res[:,0] = X2P*np.cos(sol["theta2"]) - Y2P*np.sin(sol["theta2"]) + sol["x2"]
P0_res[:,1] = X2P*np.sin(sol["theta2"]) + Y2P*np.cos(sol["theta2"]) + sol["y2"]
P1_res[:,0] = X3P*np.cos(sol["theta3"]) - Y3P*np.sin(sol["theta3"]) + sol["x3"]
P1_res[:,1] = X3P*np.sin(sol["theta3"]) + Y3P*np.cos(sol["theta3"]) + sol["y3"]
P2_res[:,0] = X4P*np.cos(sol["theta4"]) - Y4P*np.sin(sol["theta4"]) + sol["x4"]
P2_res[:,1] = X4P*np.sin(sol["theta4"]) + Y4P*np.cos(sol["theta4"]) + sol["y4"]
P3_res[:,0] = X6P*np.cos(sol["theta6"]) - Y6P*np.sin(sol["theta6"]) + sol["x6"]
P3_res[:,1] = X6P*np.sin(sol["theta6"]) + Y6P*np.cos(sol["theta6"]) + sol["y6"]


# Beta angle
fig, ax = plt.subplots()
plt.plot(t,P0_res[:,0],'k-',t,P0_res[:,1],'k--')
plt.plot(t,P1_res[:,0],'k-',t,P1_res[:,1],'k--')
plt.plot(t,P2_res[:,0],'k-',t,P2_res[:,1],'k--')
plt.plot(t,P3_res[:,0],'k-',t,P3_res[:,1],'k--')
plt.legend([r'$x_P$',r'$y_P$'],fontsize=taille)
plt.title(r'$x_P(t) \mkern9mu and \mkern9mu y_P(t)$',size=taille)
plt.xlabel(r"$t \mkern9mu (s)$",fontsize=taille)
plt.ylabel(r'$Position \mkern9mu (m)$',size=taille)
plt.xlim([0,duration])
plt.minorticks_on()
plt.grid(which="major",ls="-")
plt.grid(which="minor",ls=":")
plt.pause(0.1)
plt.tight_layout()


Theta1=np.fmod(sol["theta1"],2*np.pi)
for i in range(Theta1.size):
    if Theta1[i] >= np.pi:
        Theta1[i] -= 2*np.pi
fig, ax = plt.subplots()
plt.plot(t,Theta1,'k.')
plt.title(r'$\beta(t)$',size=taille)
plt.xlabel(r"$t \mkern9mu (s)$",fontsize=taille)
plt.ylabel(r'$Angle \mkern9mu (rad)$',size=taille)
plt.xlim([0,duration])
plt.ylim([-np.pi,np.pi])
plt.minorticks_on()
plt.grid(which="major",ls="-")
plt.grid(which="minor",ls=":")
plt.pause(0.1)
plt.tight_layout()



# numeric drift
fig, ax = plt.subplots()
plt.plot(t,drift)
plt.title("Connections' power",size=taille)
plt.xlabel(r"$t \mkern9mu (s)$",fontsize=taille)
plt.ylabel(r"$ \abs{ \dot{\lambda}(t) \cdot \left( C(\dot{q},q) \dot{q}(t) \right)} \mkern9mu (W)$",size=taille)
plt.xlim([0,duration])
plt.minorticks_on()
plt.grid(which="major",ls="-")
plt.grid(which="minor",ls=":")
plt.pause(0.1)
plt.tight_layout()



plt.show()
