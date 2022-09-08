# -*- coding: utf-8 -*-
'''
Created on Thu Sep 12 14:22:01 2019

@author: rlecuyer
'''
import numpy as np
import string

class model:
    
    def __init__(self,nb_solid):
        """
        nb_solid: number of solids of the model
        """
        self.n = 3*nb_solid
        self.m = 0
        self.B=[]
        self.C=[]
        self.n_spring = 0 # number of spring
        #######################################################################
        # Creation of lists for parameters 
        #######################################################################
        self.parameters_liaisons = []
        self.parameters_solids = []
        self.parameters_efforts_internes = []
        self.parameters_efforts_externes = []
        self.parameters_efforts_pto = []
        # mass and inertia
        for i in range(nb_solid):
            self.parameters_solids.append('m{}'.format(i+1))
            self.parameters_solids.append('I{}'.format(i+1))
        #######################################################################
        # Creation of lists for writing
        #######################################################################
        self.list_B=[]
        self.list_C=[]
        self.list_parameters=[]
        self.list_variables=[]
        self.list_forces=[]
        
    def _fill_list_B(self):
        for i in range(len(self.B)):
            char = '    '
            for j in range(self.n):
                if self.B[i][j] != " 0.0": 
                    char += 'B[{:2d},{:2d}]='.format(i,j) + str(self.B[i][j]) + '; '
            self.list_B.append(char + '\n')
        
    def _fill_list_C(self):
        for i in range(len(self.B)):
            char = '    '
            for j in range(self.n):
                if self.C[i][j] != " 0.0": 
                    char += 'C[{:2d},{:2d}]='.format(i,j) + str(self.C[i][j]) + '; '
            self.list_C.append(char + '\n')
            
    def _fill_list_parameters(self): 
    # Parameters of solids
        self.list_parameters.append("    "+"# parameters of solids"+"\n")
        for i in range(len(self.parameters_solids)):
            char = '    ' + 'cdef double {} = c_parameters["{}"]'.format(self.parameters_solids[i],self.parameters_solids[i]) + '\n'
            self.list_parameters.append(char)
        self.list_parameters.append("\n")
    # Parameters of connections 
        self.list_parameters.append("    "+"# parameters of connections"+"\n")       
        for i in range(len(self.parameters_liaisons)):
            char = '    ' + 'cdef double {} = c_parameters["{}"]'.format(self.parameters_liaisons[i][0],self.parameters_liaisons[i][0]) + '\n'
            self.list_parameters.append(char)
        self.list_parameters.append("\n")
    # Parameters of intern forces
        self.list_parameters.append("    "+"# parameters of intern forces"+"\n")       
        for i in range(len(self.parameters_efforts_internes)):
            char = '    ' + 'cdef double {} = c_parameters["{}"]'.format(self.parameters_efforts_internes[i][0],self.parameters_efforts_internes[i][0]) + '\n'
            self.list_parameters.append(char)
        self.list_parameters.append("\n")
    # Parameters of extern forces
        self.list_parameters.append("    "+"# parameters of extern forces"+"\n")       
        for i in range(len(self.parameters_efforts_externes)):
            char = '    ' + 'cdef double {} = c_parameters["{}"]'.format(self.parameters_efforts_externes[i][0],self.parameters_efforts_externes[i][0]) + '\n'
            self.list_parameters.append(char)
        self.list_parameters.append("\n")


            
    def _write_parameters(self): 
        f=open("input_parameters.txt","w+")
        # number of degrees of freedom
        f.write('parameters["n"] = {} '.format(self.n))
        f.write('\n')
        # number of constraints
        f.write('parameters["m"] = {} '.format(self.m))
        f.write('\n')
        f.write("# parameters of solids"+"\n")
        for i in range(len(self.parameters_solids)):
            f.write('parameters["{}"] = {} '.format(self.parameters_solids[i],self.parameters_solids[i]))
            f.write('\n')
        f.write('\n')
        f.write("# parameters of connections"+"\n")
        for i in range(len(self.parameters_liaisons)):
            f.write('parameters["{}"] = {} '.format(self.parameters_liaisons[i][0],self.parameters_liaisons[i][1]))
            f.write('\n')
        f.write('\n')
        f.write("# parameters of intern forces"+"\n")
        for i in range(len(self.parameters_efforts_internes)):
            f.write('parameters["{}"] = {} '.format(self.parameters_efforts_internes[i][0],self.parameters_efforts_internes[i][1]))
            f.write('\n')
        f.write('\n')
        f.write("# parameters of extern forces"+"\n")
        for i in range(len(self.parameters_efforts_externes)):
            f.write('parameters["{}"] = {} '.format(self.parameters_efforts_externes[i][0],self.parameters_efforts_externes[i][1]))
            f.write('\n')
        f.write('\n')
        f.close()
        
            
    def _fill_list_variables(self): 
    # Kinematic variables      
        for i in range(int(self.n/3)):
            char1 = '    ' + 'cdef double dx{}_0=X0[{}], dy{}_0=X0[{}], dtheta{}_0=X0[{}]'.format(i+1,3*i,i+1,3*i+1,i+1,3*i+2) + '\n'
            char2 = '    ' + 'cdef double x{}_0=X0[{}], y{}_0=X0[{}], theta{}_0=X0[{}]'.format(i+1,3*i+self.n,i+1,3*i+self.n+1,i+1,3*i+self.n+2) + '\n'
            char3 = '    ' + 'cdef double dx{}=X[{}], dy{}=X[{}], dtheta{}=X[{}]'.format(i+1,3*i,i+1,3*i+1,i+1,3*i+2) + '\n'
            char4 = '    ' + 'cdef double x{}=X[{}], y{}=X[{}], theta{}=X[{}]'.format(i+1,3*i+self.n,i+1,3*i+self.n+1,i+1,3*i+self.n+2) + '\n'
            self.list_variables.append(char1)
            self.list_variables.append(char2)
            self.list_variables.append(char3)
            self.list_variables.append(char4)
            self.list_variables.append('\n')

            
    def write_model(self,nom="model.pyx"):
        """
        Read Matrix B and C. 
        Read the cython template cython.
        Concatenate + compilation
        """
        with open("template.pyx",'r') as fid:
            raw=fid.readlines()
            
        self._write_parameters()
        self._fill_list_B()
        self._fill_list_C()
        self._fill_list_parameters()
        self._fill_list_variables()
        i1=0
        i2=0
        with open(nom+".pyx", "w+") as f:
            for i, line in enumerate(raw):
                if "#Insertion_1" in line:
                    i1=i
                if "#Insertion_2" in line:
                    i2=i
            f.writelines(raw[:i1])
            # Variables
            f.writelines("    "+"# Kinematic variables"+"\n")
            f.writelines(self.list_variables)
            f.write('\n')
            # Parameters
            f.writelines("    "+"# parameters"+"\n"+"\n")
            f.writelines(self.list_parameters)
            # Matrix B
            f.writelines("    "+"# Matrix B"+"\n")
            f.writelines(self.list_B)
            f.write('\n')
            # Matrix C
            f.writelines("    "+"# Matrix C"+"\n")
            f.writelines(self.list_C)
            f.write('\n')
            # Second part of the writing
            f.writelines(raw[i1+1:i2])
            # Forces
            f.writelines(self.list_forces)
            f.write('\n')
            # Ending
            f.writelines(raw[i2+1:])
                
    def compile_model(self):
        
        import os
        os.system('python setup.py install --user')
        
        
###############################################################################
# Mechanical connections
###############################################################################
        
        
    def add_hinge(self,i,j,p):
        """
        i: numéro of the first solid
        j: numéro of the second solid
        p: point's nale ex: p='A'
        i et j > 0
        
        Distances are parameters         
        """
        self.m += 2
        temp_B1=[' 0.0' for i in range(self.n)]
        temp_B2=[' 0.0' for i in range(self.n)]
        temp_C1=[' 0.0' for i in range(self.n)]
        temp_C2=[' 0.0' for i in range(self.n)]
        #######################################################################
        # C matrix
        #######################################################################
        
        if i!=0:
            #ith triplet for each equation
            temp_C1[3*(i-1):3*i]=[' 1.0', ' 0.0', '-sin(theta{})*X{}{} - cos(theta{})*Y{}{}'.format(i,i,p,i,i,p)]
            temp_C2[3*(i-1):3*i]=[' 0.0', ' 1.0', ' cos(theta{})*X{}{} - sin(theta{})*Y{}{}'.format(i,i,p,i,i,p)]
        else: 
            temp_C1[3*(i-1):3*i]=[' 0.0', ' 0.0', ' 0.0']
            temp_C2[3*(i-1):3*i]=[' 0.0', ' 0.0', ' 0.0']
            
        #jth triplet for each equation
        temp_C1[3*(j-1):3*j]=['-1.0', ' 0.0', ' sin(theta{})*X{}{} + cos(theta{})*Y{}{}'.format(j,j,p,j,j,p)]
        temp_C2[3*(j-1):3*j]=[' 0.0', '-1.0', '-cos(theta{})*X{}{} + sin(theta{})*Y{}{}'.format(j,j,p,j,j,p)]
            
        #######################################################################
        # B matrix = dC/dt
        #######################################################################
        if i!=0:
            #ith triplet for each equation
            temp_B1[3*(i-1):3*i]=[' 0.0', ' 0.0', '-dtheta{}*cos(theta{})*X{}{} + dtheta{}*sin(theta{})*Y{}{}'.format(i,i,i,p,i,i,i,p)]
            temp_B2[3*(i-1):3*i]=[' 0.0', ' 0.0', '-dtheta{}*sin(theta{})*X{}{} - dtheta{}*cos(theta{})*Y{}{}'.format(i,i,i,p,i,i,i,p)]
        else: 
            temp_B1[3*(i-1):3*i]=[' 0.0', ' 0.0', ' 0.0']
            temp_B2[3*(i-1):3*i]=[' 0.0', ' 0.0', ' 0.0']
            
        #jth triplet for each equation
        temp_B1[3*(j-1):3*j]=[' 0.0', ' 0.0', ' dtheta{}*cos(theta{})*X{}{} - dtheta{}*sin(theta{})*Y{}{}'.format(j,j,j,p,j,j,j,p)]
        temp_B2[3*(j-1):3*j]=[' 0.0', ' 0.0', ' dtheta{}*sin(theta{})*X{}{} + dtheta{}*cos(theta{})*Y{}{}'.format(j,j,j,p,j,j,j,p)]
        
        self.B.append(temp_B1)
        self.B.append(temp_B2)
        self.C.append(temp_C1)
        self.C.append(temp_C2)
        
        if i!=0:
            self.parameters_liaisons.append(['X{}{}'.format(i,p),' ({}[0]-x{}_0)*cos(theta{}_0)+({}[-1]-y{}_0)*sin(theta{}_0)'.format(p,i,i,p,i,i)])
            self.parameters_liaisons.append(['Y{}{}'.format(i,p),'-({}[0]-x{}_0)*sin(theta{}_0)+({}[-1]-y{}_0)*cos(theta{}_0)'.format(p,i,i,p,i,i)])
        self.parameters_liaisons.append(['X{}{}'.format(j,p),' ({}[0]-x{}_0)*cos(theta{}_0)+({}[-1]-y{}_0)*sin(theta{}_0)'.format(p,j,j,p,j,j)])
        self.parameters_liaisons.append(['Y{}{}'.format(j,p),'-({}[0]-x{}_0)*sin(theta{}_0)+({}[-1]-y{}_0)*cos(theta{}_0)'.format(p,j,j,p,j,j)])
        
        
        
###############################################################################
# Intern forces
###############################################################################
    
    def add_fixed_spring(self,i,Pi,P0,stiffness=None,zero_length_spring=None):
        """
        i: number of the first solid
        """
        alphabet = string.ascii_lowercase
        lx = "lx{}".format(alphabet[self.n_spring])
        ly = "ly{}".format(alphabet[self.n_spring])
        l = "l{}".format(alphabet[self.n_spring])
        if stiffness==None:
            stiffness="k{}".format(alphabet[self.n_spring])
        if zero_length_spring==None:
            zero_length_spring="l0{}".format(alphabet[self.n_spring])
        self.n_spring += 1
        
        self.list_forces.append("\n")
        self.list_forces.append("    "+"# Effort du ressort"+ "\n")
        # Definition of length
        char1 = '    ' + 'cdef double {} = x{}+X{}{}*cos(theta{})-Y{}{}*sin(theta{})-x{}'.format(lx,i,i,Pi,i,i,Pi,i,P0) + "\n"
        char2 = '    ' + 'cdef double {} = y{}+X{}{}*sin(theta{})+Y{}{}*cos(theta{})-y{}'.format(ly,i,i,Pi,i,i,Pi,i,P0) + "\n"
        char3 = '    ' + 'cdef double {} = ({}**2 + {}**2 )**0.5'.format(l,lx,ly) + "\n"
        self.list_forces.append(char1)  
        self.list_forces.append(char2) 
        self.list_forces.append(char3)  
        # Definifion of forces
        char1 = '    ' + 'ftot[{}] += {} * -{}*({}-{})/{}'.format(3*(i-1),lx,stiffness,l,zero_length_spring,l) + "\n"
        char2 = '    ' + 'ftot[{}] += {} * -{}*({}-{})/{}'.format(3*(i-1)+1,ly,stiffness,l,zero_length_spring,l) + "\n"
        char3 = '    ' + 'ftot[{}] += ((X{}{}*cos(theta{})-Y{}{}*sin(theta{}))*{} '.format(3*(i-1)+2,i,Pi,i,i,Pi,i,ly)
        char3 += ' - (X{}{}*sin(theta{})+Y{}{}*cos(theta{}))*{})*-{}*({}-{})/{}'.format(i,Pi,i,i,Pi,i,lx,stiffness,l,zero_length_spring,l) + "\n"
        
        self.list_forces.append(char1)  
        self.list_forces.append(char2) 
        self.list_forces.append(char3)
        
        self.parameters_efforts_externes.append(['X{}{}'.format(i, Pi),'({}[0]-x{}_0)*cos(theta{}_0)+({}[1]-y{}_0)*sin(theta{}_0)'.format(Pi,i,i,Pi,i,i)])
        self.parameters_efforts_externes.append(['Y{}{}'.format(i, Pi),'-({}[0]-x{}_0)*sin(theta{}_0)+({}[1]-y{}_0)*cos(theta{}_0)'.format(Pi,i,i,Pi,i,i)])
        self.parameters_efforts_externes.append(['x{}'.format(P0),'{}[0]'.format(P0)])
        self.parameters_efforts_externes.append(['y{}'.format(P0),'{}[1]'.format(P0)])
        self.parameters_efforts_externes.append(['{}'.format(stiffness),'{}'.format(stiffness)])
        self.parameters_efforts_externes.append(['{}'.format(zero_length_spring),'{}'.format(zero_length_spring)])
        
        

