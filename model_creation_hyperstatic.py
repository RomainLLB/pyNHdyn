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

#Hyperstatisme
mod.add_hinge(3,4,'P4')
mod.add_hinge(3,6,'P5')
mod.add_hinge(4,6,'P6')


# Add the spring
mod.add_fixed_spring(3,'D','C','ka','l0a')

# Write and compile the model
mod.write_model(nom='benchmark')
mod.compile_model()
