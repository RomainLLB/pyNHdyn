cimport cython
cimport scipy.linalg.cython_lapack
from libc.stdlib cimport malloc, free
from libc.math cimport cos, sin, exp, tan, pi, atan, tanh, fabs, acos, fmod
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from tqdm import trange, tqdm
import scipy
import h5py
	

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef csolve_dgesv(double[::1,:] A , double[::1] B):
	
    cdef int N = A.shape[1]
    cdef int NRHS = 1
    cdef int LDA = A.shape[0]
    cdef int LDB = A.shape[0]
    cdef int info = 0
	
    cdef int* piv_pointer = <int*>malloc(sizeof(int)*N)
    if not piv_pointer:
        raise MemoryError()	

    try:		
        scipy.linalg.cython_lapack.dgesv(&N,&NRHS,&A[0,0],&LDA,piv_pointer,&B[0],&LDB,&info)	
        if info!=0:
            raise NameError('error in dgesv')
    except NameError:
        raise NameError('error in dgesv')

    finally:
        free(piv_pointer)
        
        
        
       
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)       
cdef csolve_dgelsd(double[::1,:] A , double[::1] B):
	
    cdef int M = A.shape[0]
    cdef int N = A.shape[0]
    cdef int NRHS = 1
    cdef int LDA = A.shape[0]
    cdef int LDB = A.shape[0]
    cdef double* s = <double*>malloc(sizeof(double)*N)
    cdef double rcond = -1
    cdef double wkopt = 0
    cdef int rank = 0
    cdef int lwork = -1
    
    cdef int MINMN = np.min([N,M])
    cdef int SMLSIZ = 25
    cdef int NLVL = np.max([0,int(np.log2(np.double(MINMN)/(np.double(SMLSIZ)+1)))+1])
    cdef int liwork = np.max([1,3*MINMN*NLVL+11*MINMN])
    cdef int* iwork = <int*>malloc(sizeof(int)*liwork)
    cdef int info = 0   
  
    
    if not s:
        raise MemoryError()	
        
    if not iwork:
        raise MemoryError()	
    
    # Optimal size for lwork
    try:		
        scipy.linalg.cython_lapack.dgelsd(&M,&N,&NRHS,&A[0,0],&LDA,&B[0],&LDB,s,&rcond,&rank,&wkopt,&lwork,iwork,&info)	
        if info!=0:
            raise NameError('error in dgelsd')
    except NameError:
        raise NameError('error in dgelsd')
    
    lwork = int(wkopt)
    cdef double* work = <double*>malloc(sizeof(double)*lwork)  
    
    # Solve AX=B knowing lwork
    try:		
        scipy.linalg.cython_lapack.dgelsd(&M,&N,&NRHS,&A[0,0],&LDA,&B[0],&LDB,s,&rcond,&rank,work,&lwork,iwork,&info)	
        if info!=0:
            raise NameError('error in dgelsd')
    except NameError:
        raise NameError('error in dgelsd')

    finally:
        free(s)
        free(iwork)
        free(work)




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef csolve_dgbsv(double[::1,:] A, double[::1] B, dict c_parameters):
    
    cdef int[::1] perm = c_parameters["perm"]
    cdef int KL = c_parameters["KL"]
    cdef int KU = c_parameters["KU"]
    cdef int N = A.shape[0], i=0, j=0
    cdef int NRHS = 1
    cdef int LDB = N
    cdef int info = 0   
    
    cdef double[::1,:] Ap = np.zeros([N,N],order="F")
    cdef double[::1] Bp = np.zeros([N])
    cdef double[::1,:] AB = np.zeros([2*KL+KU+1,N],order="F")   
    cdef int LDAB = 2*KL+KU+1
    
    cdef int* piv_pointer = <int*>malloc(sizeof(int)*N)

    # permutation
    for i in range(N):
        Bp[i]=B[perm[i]]
        for j in range(N):
            Ap[i,j]=A[perm[i],perm[j]]
    
    # band format
    for j in range(1,N+1):   
        for i in range(max(1,j-KU),min(N,j+KL)+1):
            AB[KU + KL + i - j , j-1] = Ap[i-1,j-1]

    try:
        scipy.linalg.cython_lapack.dgbsv(&N,&KL,&KU,&NRHS,&AB[0,0],&LDAB,piv_pointer,&Bp[0],&LDB,&info)	
        if info!=0:
            raise NameError('error in dgbsv')
    except NameError:
        raise NameError('error in dgbsv')
        
    finally:
        free(piv_pointer)
        
    for i in range(N):
        B[perm[i]] = Bp[i]
    


    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef csolve_dgels(double[::1,:] A , double[::1] B):
    
    cdef char* TRANS = 'N'
    cdef int N = A.shape[1]
    cdef int NRHS = 1
    cdef int LDA = A.shape[0]
    cdef int LDB = A.shape[0]
    cdef int lwork = 2*N
    cdef int info = 0
	
    cdef double* work = <double*>malloc(sizeof(double)*lwork)
    if not work:
        raise MemoryError()	

    try:		
        scipy.linalg.cython_lapack.dgels(TRANS,&N,&N,&NRHS,&A[0,0],&LDA,&B[0],&LDB,work,&lwork,&info)	
        if info!=0:
            raise NameError('error in dgesv')
    except NameError:
        raise NameError('error in dgesv')

    finally:
        free(work)





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cstate_space_function(double ti,double[::1] X0, double[::1] X, double[::1] dX, dict c_parameters, double[:,::1] B, double[:,::1] C, double[:,::1] tB, double[:,::1] tC,
    double[::1,:] A1, double[::1,:] A2, double[::1] ftot):
	
	# Simulation entries
    cdef double[:,::1] MA = c_parameters["MA"]
    cdef double[:,::1] Inn = c_parameters["Inn"]
    cdef int n=c_parameters["n"],m=c_parameters["m"]
    cdef int i=0, j=0, k=0, l=0
    cdef double rho=1025.,g=9.81	
    
	     

	# Variables

	#Insertion_1
	

	###########################################################################
	# Forces
	###########################################################################


    ftot[:]=0.0
	
    
    #torque 
    ftot[2] += c_parameters["torque"]
    

    
	#Insertion_2
    
	
    # Creation of matrix A1 and A2
    for i in range(m):
        for j in range(n):
            tB[j,i]=B[i,j]
            tC[j,i]=C[i,j]
	
    # matrix A1 
    A1[:,:]=0.0 
    A1[0:n,0:n]=MA[:,:]
    A1[0:n,2*n:2*n+m]=tC[:,:]
    A1[n:2*n,n:2*n]=Inn[:,:]
    A1[2*n:2*n+m,0:n]=C[:,:]
    

    A2[n:2*n,0:n]=Inn[:,:]
    A2[2*n:2*n+m,0:n]=B[:,:]
    for i in range(2*n,2*n+m):
        for j in range(0,n):
            A2[i,j]=-A2[i,j]
    
	#dXdyn
    for i in range(2*n+m):
        dX[i]=ftot[i]
        for j in range(2*n+m):
            dX[i]+=A2[i,j]*X[j]
            
    if c_parameters["solver"]=='dgesv':
        csolve_dgesv(A1[:,:],dX[0:2*n+m])
    elif c_parameters["solver"]=='dgelsd':
        csolve_dgelsd(A1[:,:],dX[0:2*n+m])
    elif c_parameters["solver"]=="dgbsv":
        csolve_dgbsv(A1[:,:],dX[0:2*n+m],c_parameters)
    elif c_parameters["solver"]=="dgels":
        csolve_dgels(A1[:,:],dX[0:2*n+m])
    
    
    
    

    


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RK4(t,X0,parameters,solver='dgesv'):
    # Definition of reals and integers
    cdef int n=parameters["n"], m=parameters["m"]
    cdef double ti=0
    cdef int ntot=X0.shape[0]
    cdef int nt=t.shape[0]
    cdef int i,j,k, ii, jj
    cdef double h
    # Definition of matrix and arrays
    Xsol=np.zeros([ntot,nt])
    dXsol=np.zeros([ntot,nt])
    Xsol[:,0]= X0[:]
    X1=X0.copy()
    k1=np.zeros([ntot])
    k2=np.zeros([ntot])
    k3=np.zeros([ntot])
    k4=np.zeros([ntot])
    X2=np.zeros([ntot])
    X3=np.zeros([ntot])
    X4=np.zeros([ntot])		
    A1=np.zeros([2*n+m,2*n+m],order="F")	
    A2=np.zeros([2*n+m,2*n+m],order="F")	
    ftot=np.zeros([2*n+m])	
    # Definition of memory_views 
    cdef double[:,::1] Xsol_v=Xsol
    cdef double[:,::1] dXsol_v=dXsol
    cdef double[::1] k1_v=k1 	
    cdef double[::1] k2_v=k2
    cdef double[::1] k3_v=k3
    cdef double[::1] k4_v=k4
    cdef double[::1] X2_v=X2
    cdef double[::1] X3_v=X3
    cdef double[::1] X4_v=X4	
    cdef double[::1] X0_v=X0
    cdef double[::1] X1_v=X1
    cdef double[::1,:] A1_v=A1
    cdef double[::1,:] A2_v=A2
    cdef double[::1] ftot_v=ftot
    
    # Definition of constraint matrix 
    B = np.zeros([m,n])
    C = np.zeros([m,n])
    tB = np.zeros([n,m])
    tC = np.zeros([n,m])
    cdef double[:,::1] B_v = B
    cdef double[:,::1] C_v = C
    cdef double[:,::1] tB_v = tB
    cdef double[:,::1] tC_v = tC
    
    # numerical drift
    drift = np.zeros([m,nt])
    cdef double [:,::1] drift_v = drift
    
    # sparse routine bor banded matrix
    if solver=="dgbsv":
        my_file = h5py.File("sparse_data.h5","r")
        parameters["KL"] = my_file["KL"][()]
        parameters["KU"] = my_file["KU"][()]
        parameters["perm"] = my_file["perm"][()]
        my_file.close()
    
    parameters["solver"]=solver
   
    # c_definition of the parameters dictionnary
    cdef dict c_parameters = parameters
    
    # Beginning of the RK4	
    with tqdm(total=nt, desc='Calculation in progress...') as pbar:
        for i in range(nt-1):
        
            h=t[i+1]-t[i]
            ti=t[i]
            cstate_space_function(ti, X0_v, X1_v, k1_v, c_parameters, B_v, C_v, tB_v, tC_v, A1_v, A2_v, ftot_v)
            for ii in range(m):
                for jj in range(n):
                    drift_v[ii,i]+=C_v[ii,jj]*X1_v[jj]
            dXsol_v[:,i] = k1_v[:]
            
            for k in range(ntot):
                X2_v[k]=X1_v[k]+h/2.0*k1_v[k]
            ti=t[i]+0.5*h
            cstate_space_function(ti, X0_v, X2_v, k2_v, c_parameters, B_v, C_v, tB_v, tC_v, A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X3_v[k]=X1_v[k]+h/2.0*k2_v[k]
            ti=t[i]+0.5*h
            cstate_space_function(ti, X0_v, X3_v, k3_v, c_parameters, B_v, C_v, tB_v, tC_v, A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X4_v[k]=X1_v[k]+h*k3_v[k]
            ti=t[i]+h
            cstate_space_function(ti, X0_v, X4_v, k4_v, c_parameters, B_v, C_v, tB_v, tC_v, A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X1_v[k]+=h/6.0*(k1_v[k]+2.0*k2_v[k]+2.0*k3_v[k]+k4_v[k])
            Xsol_v[:,i+1]=X1_v[:]
            
            pbar.update(1)
            
        ti=t[nt-1]
        cstate_space_function(ti, X0_v, X1_v,k1_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)
        for ii in range(m):
            for jj in range(n):
                drift_v[ii,nt-1]+=C_v[ii,jj]*X1_v[jj]
        dXsol_v[:,nt-1] = k1_v[:]          
        pbar.update(1)
		
    return(dXsol, Xsol,drift)


# The same with RK45
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RK45(t,X0,parameters,solver='dgesv'):
    cdef int n=parameters["n"], m=parameters["m"]
    cdef double ti=0
    cdef int ntot=X0.shape[0]
    cdef int nt=t.shape[0]
    cdef int i,j,k, ii, jj
    cdef double h
    Xsol=np.zeros([ntot,nt])
    dXsol=np.zeros([ntot,nt])
    Xsol[:,0]= X0[:]
    X1=X0.copy()
    k1=np.zeros([ntot])
    k2=np.zeros([ntot])
    k3=np.zeros([ntot])
    k4=np.zeros([ntot])
    k5=np.zeros([ntot])
    k6=np.zeros([ntot])
    X2=np.zeros([ntot])
    X3=np.zeros([ntot])
    X4=np.zeros([ntot])
    X5=np.zeros([ntot])	
    X6=np.zeros([ntot])	    
    A1=np.zeros([2*n+m,2*n+m], order="F")	
    A2=np.zeros([2*n+m,2*n+m], order="F")	
    ftot=np.zeros([2*n+m])	
    cdef double[:,::1] Xsol_v=Xsol
    cdef double[:,::1] dXsol_v=dXsol
    cdef double[::1] k1_v=k1 	
    cdef double[::1] k2_v=k2
    cdef double[::1] k3_v=k3
    cdef double[::1] k4_v=k4
    cdef double[::1] k5_v=k5
    cdef double[::1] k6_v=k6
    cdef double[::1] X2_v=X2
    cdef double[::1] X3_v=X3
    cdef double[::1] X4_v=X4
    cdef double[::1] X5_v=X5
    cdef double[::1] X6_v=X6	
    cdef double[::1] X0_v=X0
    cdef double[::1] X1_v=X1
    cdef double[::1,:] A1_v=A1
    cdef double[::1,:] A2_v=A2
    cdef double[::1] ftot_v=ftot
    
    B = np.zeros([m,n])
    C = np.zeros([m,n])
    tB = np.zeros([n,m])
    tC = np.zeros([n,m])
    cdef double[:,::1] B_v = B
    cdef double[:,::1] C_v = C
    cdef double[:,::1] tB_v = tB
    cdef double[:,::1] tC_v = tC
    
    drift = np.zeros([m,nt])
    cdef double [:,::1] drift_v = drift
    
    if solver=="dgbsv":
        my_file = h5py.File("sparse_data.h5","r")
        parameters["KL"] = my_file["KL"][()]
        parameters["KU"] = my_file["KU"][()]
        parameters["perm"] = my_file["perm"][()]
        my_file.close()
    
    parameters["solver"]=solver
    cdef dict c_parameters = parameters
    
    with tqdm(total=nt, desc='Calculation in progress...') as pbar:
        for i in range(nt-1):
        
            h=t[i+1]-t[i]
            ti=t[i]
            cstate_space_function(ti, X0_v, X1_v,k1_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)
            for ii in range(m):
                for jj in range(n):
                    drift_v[ii,i]+=C_v[ii,jj]*X1_v[jj]
            dXsol_v[:,i] = k1_v[:]
            
            for k in range(ntot):
                X2_v[k]=X1_v[k]+h/5.*k1_v[k]
            ti=t[i]+h/5.
            cstate_space_function(ti, X0_v, X2_v, k2_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X3_v[k]=X1_v[k]+h*(3./40.*k1_v[k]+9./40.*k2_v[k])
            ti=t[i]+3.*h/10.
            cstate_space_function(ti, X0_v, X3_v,k3_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X4_v[k] =X1_v[k]+h*(44./45.*k1_v[k]-56./15.*k2_v[k]+32./9.*k3_v[k])
            ti=t[i]+4.*h/5.
            cstate_space_function(ti, X0_v, X4_v,k4_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)
            
            for k in range(ntot):
                X5_v[k]=X1_v[k]+h*(19372./6561.*k1_v[k]-25360./2187.*k2_v[k]+64448./6561.*k3_v[k]-212./729.*k4_v[k])
            ti=t[i]+8.*h/9.
            cstate_space_function(ti, X0_v, X5_v,k5_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)
            
            for k in range(ntot):
                X6_v[k]=X1_v[k]+h*(9017./3168.*k1_v[k]-355./33.*k2_v[k]+46732./5247.*k3_v[k]+49./176.*k4_v[k]-5103./18656.*k5_v[k])
            ti=t[i]+h
            cstate_space_function(ti, X0_v, X6_v,k6_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)

            for k in range(ntot):
                X1_v[k]+=h*(35./384.*k1_v[k]+500./1113.*k3_v[k]+125./192.*k4_v[k]-2187./6784.*k5_v[k]+11./84.*k6_v[k])
            Xsol_v[:,i+1] = X1_v[:]
            
            pbar.update(1)
            
        ti=t[nt-1]
        cstate_space_function(ti, X0_v, X1_v,k1_v,c_parameters,B_v,C_v,tB_v,tC_v,  A1_v, A2_v, ftot_v)
        for ii in range(m):
            for jj in range(n):
                drift_v[ii,nt-1]+=C_v[ii,jj]*X1_v[jj]
        dXsol_v[:,nt-1] = k1_v[:]    
        pbar.update(1)
		
    return(dXsol, Xsol, drift)	
