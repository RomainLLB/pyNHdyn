# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:28:36 2019

@author: rlecuyer
"""
import numpy as np
import os
from class_model import model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
import h5py
import matplotlib.pyplot as plt

#######################################################################
# Example of model construction
#######################################################################                         

# Initialization
mod=model(7) # 7 solids 

# Add mechanical connections
mod.add_hinge(0,5,'A1')
mod.add_hinge(0,7,'A2')

mod.add_hinge(0,1,'O')
mod.add_hinge(0,3,'B')
mod.add_hinge(4,5,'E')
mod.add_hinge(6,7,'F')
mod.add_hinge(1,2,'G')

mod.add_hinge(2,3,'P1')
mod.add_hinge(2,4,'P2')
mod.add_hinge(2,6,'P3')



# Add the spring
mod.add_fixed_spring(3,'D','C','ka','l0a')

# Write and compile the model
mod.write_model(nom='benchmark')
mod.compile_model()



##############################################################################
# Band matrix for dgbsv routine
##############################################################################

# matrix C
C=np.zeros([mod.m,mod.n],dtype=np.int32)
for i in range(mod.m):
    for j in range(mod.n):
        if mod.C[i][j] != " 0.0":
            C[i,j]=1
            
# left array
A=np.zeros([2*mod.n+mod.m,2*mod.n+mod.m])
for i in range(2*mod.n):
    A[i,i] = 1
A[0:mod.n,2*mod.n:2*mod.n+mod.m] = C.T
A[2*mod.n:2*mod.n+mod.m,0:mod.n] = C

# permutation 
perm = reverse_cuthill_mckee(csr_matrix(A))
Ap=np.zeros([2*mod.n+mod.m,2*mod.n+mod.m])
for i in range(2*mod.n+mod.m):
    for j in range(2*mod.n+mod.m):
        Ap[i,j]=A[perm[i],perm[j]]               
nz_i = Ap.nonzero()[0] # row indices of non zeros values
nz_j = Ap.nonzero()[1] # column indices of non zeros values

# number of bands
upper_diag=[]
lower_diag=[]
for ii in range(np.count_nonzero(Ap)):
    i = nz_i[ii]
    j = nz_j[ii]
    if i==j:
        pass
    elif j>i:
        upper_diag.append(j-i)
    elif j<i:
        lower_diag.append(i-j)          
l = np.max(np.array(lower_diag))            
u = np.max(np.array(upper_diag))

# writing data
data_for_saving={"KL":l,
                 "KU":u,
                 "perm":np.ascontiguousarray(perm,dtype=np.int32)}

nom='sparse_data.h5'
path = os.path.abspath('')+'\\'+nom
if os.path.exists(path):
    os.remove(path)
with h5py.File(path,"a") as mon_fichier:
    for key in data_for_saving.keys():
        arr=data_for_saving[key]
        data_dir=key
        mon_fichier.create_dataset(data_dir,data=arr)   
    mon_fichier.flush()
    

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{physics}\usepackage{gensymb}')

plot_size=20
plt.rc('font', size=plot_size)

fig, ax = plt.subplots()
c = ax.pcolor(A, edgecolors='k', linewidths=0.5, cmap='Blues')
ax.set_title('Matrix A before permutation',size=plot_size)
xl = plt.xlabel('column indice',size=plot_size)
yl = plt.ylabel('row indice',size=plot_size)
ax.set_xlim(left=0,right=A.shape[0])
ax.set_ylim(bottom=A.shape[0],top=0)
ax.set_aspect(aspect=1)
plt.pause(0.1)
plt.tight_layout()


fig, ax = plt.subplots()
c = ax.pcolor(Ap, edgecolors='k', linewidths=0.5, cmap='Blues')
ax.set_title('Matrix A after permutation',size=plot_size)
xl = plt.xlabel('column indice',size=plot_size)
yl = plt.ylabel('row indice',size=plot_size)
ax.set_xlim(left=0,right=Ap.shape[0])
ax.set_ylim(bottom=Ap.shape[0],top=0)
ax.set_aspect(aspect=1)
plt.pause(0.1)
plt.tight_layout()


plt.show()
