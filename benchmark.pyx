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
cdef csolve_dgesv(double[:,::1] A , double[::1] B):
	
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
cdef csolve_dgelsd(double[:,::1] A , double[::1] B):
	
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
cdef csolve_dgbsv(double[:,::1] A, double[::1] B, dict c_parameters):
    
    cdef int[::1] perm = c_parameters["perm"]
    cdef int KL = c_parameters["KL"]
    cdef int KU = c_parameters["KU"]
    cdef int N = A.shape[0], i=0, j=0
    
    cdef double[:,::1] Ap = np.zeros([N,N])
    cdef double[::1] Bp = np.zeros([N])
    cdef double[:,::1] AB = np.zeros([2*KL+KU+1,N])   

    # permutation
    for i in range(N):
        Bp[i]=B[perm[i]]
        for j in range(N):
            Ap[i,j]=A[perm[i],perm[j]]
    
    # band format
    for j in range(1,N+1):   
        for i in range(max(1,j-KU),min(N,j+KL)+1):
            AB[KU + KL + i - j , j-1] = Ap[i-1,j-1]
    
    # solve
    scipy.linalg.lapack.dgbsv(KL, KU, AB[:,:], Bp[:], overwrite_ab=1, overwrite_b=1)
    
    for i in range(N):
        B[perm[i]] = Bp[i]
    


    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef csolve_dgels(double[:,::1] A , double[::1] B):
    
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
    double[:,::1] A1, double[:,::1] A2, double[::1] ftot):
	
	# Simulation entries
    cdef double[:,::1] MA = c_parameters["MA"]
    cdef double[:,::1] Inn = c_parameters["Inn"]
    cdef int n=c_parameters["n"],m=c_parameters["m"]
    cdef int i=0, j=0, k=0, l=0
    cdef double rho=1025.,g=9.81	
    
	     

	# Variables

    # Kinematic variables
    cdef double dx1_0=X0[0], dy1_0=X0[1], dtheta1_0=X0[2]
    cdef double x1_0=X0[21], y1_0=X0[22], theta1_0=X0[23]
    cdef double dx1=X[0], dy1=X[1], dtheta1=X[2]
    cdef double x1=X[21], y1=X[22], theta1=X[23]

    cdef double dx2_0=X0[3], dy2_0=X0[4], dtheta2_0=X0[5]
    cdef double x2_0=X0[24], y2_0=X0[25], theta2_0=X0[26]
    cdef double dx2=X[3], dy2=X[4], dtheta2=X[5]
    cdef double x2=X[24], y2=X[25], theta2=X[26]

    cdef double dx3_0=X0[6], dy3_0=X0[7], dtheta3_0=X0[8]
    cdef double x3_0=X0[27], y3_0=X0[28], theta3_0=X0[29]
    cdef double dx3=X[6], dy3=X[7], dtheta3=X[8]
    cdef double x3=X[27], y3=X[28], theta3=X[29]

    cdef double dx4_0=X0[9], dy4_0=X0[10], dtheta4_0=X0[11]
    cdef double x4_0=X0[30], y4_0=X0[31], theta4_0=X0[32]
    cdef double dx4=X[9], dy4=X[10], dtheta4=X[11]
    cdef double x4=X[30], y4=X[31], theta4=X[32]

    cdef double dx5_0=X0[12], dy5_0=X0[13], dtheta5_0=X0[14]
    cdef double x5_0=X0[33], y5_0=X0[34], theta5_0=X0[35]
    cdef double dx5=X[12], dy5=X[13], dtheta5=X[14]
    cdef double x5=X[33], y5=X[34], theta5=X[35]

    cdef double dx6_0=X0[15], dy6_0=X0[16], dtheta6_0=X0[17]
    cdef double x6_0=X0[36], y6_0=X0[37], theta6_0=X0[38]
    cdef double dx6=X[15], dy6=X[16], dtheta6=X[17]
    cdef double x6=X[36], y6=X[37], theta6=X[38]

    cdef double dx7_0=X0[18], dy7_0=X0[19], dtheta7_0=X0[20]
    cdef double x7_0=X0[39], y7_0=X0[40], theta7_0=X0[41]
    cdef double dx7=X[18], dy7=X[19], dtheta7=X[20]
    cdef double x7=X[39], y7=X[40], theta7=X[41]


    # parameters

    # parameters of solids
    cdef double m1 = c_parameters["m1"]
    cdef double I1 = c_parameters["I1"]
    cdef double m2 = c_parameters["m2"]
    cdef double I2 = c_parameters["I2"]
    cdef double m3 = c_parameters["m3"]
    cdef double I3 = c_parameters["I3"]
    cdef double m4 = c_parameters["m4"]
    cdef double I4 = c_parameters["I4"]
    cdef double m5 = c_parameters["m5"]
    cdef double I5 = c_parameters["I5"]
    cdef double m6 = c_parameters["m6"]
    cdef double I6 = c_parameters["I6"]
    cdef double m7 = c_parameters["m7"]
    cdef double I7 = c_parameters["I7"]

    # parameters of connections
    cdef double X5A1 = c_parameters["X5A1"]
    cdef double Y5A1 = c_parameters["Y5A1"]
    cdef double X7A2 = c_parameters["X7A2"]
    cdef double Y7A2 = c_parameters["Y7A2"]
    cdef double X1O = c_parameters["X1O"]
    cdef double Y1O = c_parameters["Y1O"]
    cdef double X3B = c_parameters["X3B"]
    cdef double Y3B = c_parameters["Y3B"]
    cdef double X4E = c_parameters["X4E"]
    cdef double Y4E = c_parameters["Y4E"]
    cdef double X5E = c_parameters["X5E"]
    cdef double Y5E = c_parameters["Y5E"]
    cdef double X6F = c_parameters["X6F"]
    cdef double Y6F = c_parameters["Y6F"]
    cdef double X7F = c_parameters["X7F"]
    cdef double Y7F = c_parameters["Y7F"]
    cdef double X1G = c_parameters["X1G"]
    cdef double Y1G = c_parameters["Y1G"]
    cdef double X2G = c_parameters["X2G"]
    cdef double Y2G = c_parameters["Y2G"]
    cdef double X2P1 = c_parameters["X2P1"]
    cdef double Y2P1 = c_parameters["Y2P1"]
    cdef double X3P1 = c_parameters["X3P1"]
    cdef double Y3P1 = c_parameters["Y3P1"]
    cdef double X2P2 = c_parameters["X2P2"]
    cdef double Y2P2 = c_parameters["Y2P2"]
    cdef double X4P2 = c_parameters["X4P2"]
    cdef double Y4P2 = c_parameters["Y4P2"]
    cdef double X2P3 = c_parameters["X2P3"]
    cdef double Y2P3 = c_parameters["Y2P3"]
    cdef double X6P3 = c_parameters["X6P3"]
    cdef double Y6P3 = c_parameters["Y6P3"]

    # parameters of intern forces

    # parameters of extern forces
    cdef double X3D = c_parameters["X3D"]
    cdef double Y3D = c_parameters["Y3D"]
    cdef double xC = c_parameters["xC"]
    cdef double yC = c_parameters["yC"]
    cdef double ka = c_parameters["ka"]
    cdef double l0a = c_parameters["l0a"]

    # Matrix B
    B[ 0,14]= dtheta5*cos(theta5)*X5A1 - dtheta5*sin(theta5)*Y5A1; 
    B[ 1,14]= dtheta5*sin(theta5)*X5A1 + dtheta5*cos(theta5)*Y5A1; 
    B[ 2,20]= dtheta7*cos(theta7)*X7A2 - dtheta7*sin(theta7)*Y7A2; 
    B[ 3,20]= dtheta7*sin(theta7)*X7A2 + dtheta7*cos(theta7)*Y7A2; 
    B[ 4, 2]= dtheta1*cos(theta1)*X1O - dtheta1*sin(theta1)*Y1O; 
    B[ 5, 2]= dtheta1*sin(theta1)*X1O + dtheta1*cos(theta1)*Y1O; 
    B[ 6, 8]= dtheta3*cos(theta3)*X3B - dtheta3*sin(theta3)*Y3B; 
    B[ 7, 8]= dtheta3*sin(theta3)*X3B + dtheta3*cos(theta3)*Y3B; 
    B[ 8,11]=-dtheta4*cos(theta4)*X4E + dtheta4*sin(theta4)*Y4E; B[ 8,14]= dtheta5*cos(theta5)*X5E - dtheta5*sin(theta5)*Y5E; 
    B[ 9,11]=-dtheta4*sin(theta4)*X4E - dtheta4*cos(theta4)*Y4E; B[ 9,14]= dtheta5*sin(theta5)*X5E + dtheta5*cos(theta5)*Y5E; 
    B[10,17]=-dtheta6*cos(theta6)*X6F + dtheta6*sin(theta6)*Y6F; B[10,20]= dtheta7*cos(theta7)*X7F - dtheta7*sin(theta7)*Y7F; 
    B[11,17]=-dtheta6*sin(theta6)*X6F - dtheta6*cos(theta6)*Y6F; B[11,20]= dtheta7*sin(theta7)*X7F + dtheta7*cos(theta7)*Y7F; 
    B[12, 2]=-dtheta1*cos(theta1)*X1G + dtheta1*sin(theta1)*Y1G; B[12, 5]= dtheta2*cos(theta2)*X2G - dtheta2*sin(theta2)*Y2G; 
    B[13, 2]=-dtheta1*sin(theta1)*X1G - dtheta1*cos(theta1)*Y1G; B[13, 5]= dtheta2*sin(theta2)*X2G + dtheta2*cos(theta2)*Y2G; 
    B[14, 5]=-dtheta2*cos(theta2)*X2P1 + dtheta2*sin(theta2)*Y2P1; B[14, 8]= dtheta3*cos(theta3)*X3P1 - dtheta3*sin(theta3)*Y3P1; 
    B[15, 5]=-dtheta2*sin(theta2)*X2P1 - dtheta2*cos(theta2)*Y2P1; B[15, 8]= dtheta3*sin(theta3)*X3P1 + dtheta3*cos(theta3)*Y3P1; 
    B[16, 5]=-dtheta2*cos(theta2)*X2P2 + dtheta2*sin(theta2)*Y2P2; B[16,11]= dtheta4*cos(theta4)*X4P2 - dtheta4*sin(theta4)*Y4P2; 
    B[17, 5]=-dtheta2*sin(theta2)*X2P2 - dtheta2*cos(theta2)*Y2P2; B[17,11]= dtheta4*sin(theta4)*X4P2 + dtheta4*cos(theta4)*Y4P2; 
    B[18, 5]=-dtheta2*cos(theta2)*X2P3 + dtheta2*sin(theta2)*Y2P3; B[18,17]= dtheta6*cos(theta6)*X6P3 - dtheta6*sin(theta6)*Y6P3; 
    B[19, 5]=-dtheta2*sin(theta2)*X2P3 - dtheta2*cos(theta2)*Y2P3; B[19,17]= dtheta6*sin(theta6)*X6P3 + dtheta6*cos(theta6)*Y6P3; 

    # Matrix C
    C[ 0,12]=-1.0; C[ 0,14]= sin(theta5)*X5A1 + cos(theta5)*Y5A1; 
    C[ 1,13]=-1.0; C[ 1,14]=-cos(theta5)*X5A1 + sin(theta5)*Y5A1; 
    C[ 2,18]=-1.0; C[ 2,20]= sin(theta7)*X7A2 + cos(theta7)*Y7A2; 
    C[ 3,19]=-1.0; C[ 3,20]=-cos(theta7)*X7A2 + sin(theta7)*Y7A2; 
    C[ 4, 0]=-1.0; C[ 4, 2]= sin(theta1)*X1O + cos(theta1)*Y1O; 
    C[ 5, 1]=-1.0; C[ 5, 2]=-cos(theta1)*X1O + sin(theta1)*Y1O; 
    C[ 6, 6]=-1.0; C[ 6, 8]= sin(theta3)*X3B + cos(theta3)*Y3B; 
    C[ 7, 7]=-1.0; C[ 7, 8]=-cos(theta3)*X3B + sin(theta3)*Y3B; 
    C[ 8, 9]= 1.0; C[ 8,11]=-sin(theta4)*X4E - cos(theta4)*Y4E; C[ 8,12]=-1.0; C[ 8,14]= sin(theta5)*X5E + cos(theta5)*Y5E; 
    C[ 9,10]= 1.0; C[ 9,11]= cos(theta4)*X4E - sin(theta4)*Y4E; C[ 9,13]=-1.0; C[ 9,14]=-cos(theta5)*X5E + sin(theta5)*Y5E; 
    C[10,15]= 1.0; C[10,17]=-sin(theta6)*X6F - cos(theta6)*Y6F; C[10,18]=-1.0; C[10,20]= sin(theta7)*X7F + cos(theta7)*Y7F; 
    C[11,16]= 1.0; C[11,17]= cos(theta6)*X6F - sin(theta6)*Y6F; C[11,19]=-1.0; C[11,20]=-cos(theta7)*X7F + sin(theta7)*Y7F; 
    C[12, 0]= 1.0; C[12, 2]=-sin(theta1)*X1G - cos(theta1)*Y1G; C[12, 3]=-1.0; C[12, 5]= sin(theta2)*X2G + cos(theta2)*Y2G; 
    C[13, 1]= 1.0; C[13, 2]= cos(theta1)*X1G - sin(theta1)*Y1G; C[13, 4]=-1.0; C[13, 5]=-cos(theta2)*X2G + sin(theta2)*Y2G; 
    C[14, 3]= 1.0; C[14, 5]=-sin(theta2)*X2P1 - cos(theta2)*Y2P1; C[14, 6]=-1.0; C[14, 8]= sin(theta3)*X3P1 + cos(theta3)*Y3P1; 
    C[15, 4]= 1.0; C[15, 5]= cos(theta2)*X2P1 - sin(theta2)*Y2P1; C[15, 7]=-1.0; C[15, 8]=-cos(theta3)*X3P1 + sin(theta3)*Y3P1; 
    C[16, 3]= 1.0; C[16, 5]=-sin(theta2)*X2P2 - cos(theta2)*Y2P2; C[16, 9]=-1.0; C[16,11]= sin(theta4)*X4P2 + cos(theta4)*Y4P2; 
    C[17, 4]= 1.0; C[17, 5]= cos(theta2)*X2P2 - sin(theta2)*Y2P2; C[17,10]=-1.0; C[17,11]=-cos(theta4)*X4P2 + sin(theta4)*Y4P2; 
    C[18, 3]= 1.0; C[18, 5]=-sin(theta2)*X2P3 - cos(theta2)*Y2P3; C[18,15]=-1.0; C[18,17]= sin(theta6)*X6P3 + cos(theta6)*Y6P3; 
    C[19, 4]= 1.0; C[19, 5]= cos(theta2)*X2P3 - sin(theta2)*Y2P3; C[19,16]=-1.0; C[19,17]=-cos(theta6)*X6P3 + sin(theta6)*Y6P3; 

	

	###########################################################################
	# Forces
	###########################################################################


    ftot[:]=0.0
	
    
    #torque 
    ftot[2] += c_parameters["torque"]
    

    

    # Effort du ressort
    cdef double lxa = x3+X3D*cos(theta3)-Y3D*sin(theta3)-xC
    cdef double lya = y3+X3D*sin(theta3)+Y3D*cos(theta3)-yC
    cdef double la = (lxa**2 + lya**2 )**0.5
    ftot[6] += lxa * -ka*(la-l0a)/la
    ftot[7] += lya * -ka*(la-l0a)/la
    ftot[8] += ((X3D*cos(theta3)-Y3D*sin(theta3))*lya  - (X3D*sin(theta3)+Y3D*cos(theta3))*lxa)*-ka*(la-l0a)/la

    
	
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
    A1=np.zeros([2*n+m,2*n+m])	
    A2=np.zeros([2*n+m,2*n+m])	
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
    cdef double[:,::1] A1_v=A1
    cdef double[:,::1] A2_v=A2
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
    A1=np.zeros([2*n+m,2*n+m])	
    A2=np.zeros([2*n+m,2*n+m])	
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
    cdef double[:,::1] A1_v=A1
    cdef double[:,::1] A2_v=A2
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