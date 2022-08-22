#Last update: Sep/09/19
import random
import numpy as np
from numpy import linalg
import scipy
from scipy import sparse
import time

# Note that last element of S_list is S_d

def Rotate_fun(myList,Rn):
    MaxOfList = max(myList)
    res_List = [(2**Rn)*x for x in myList[:-1]]+[MaxOfList]
    for x in res_List[:-1]:
        if x > MaxOfList:
            res_List[res_List.index(x)] = x%MaxOfList
    return res_List

def Rho_A(N, myd, Psi):
    myA_Mat_new = np.zeros((2**(N/2),2**(N/2)), dtype=complex)
    for dn in range(myd):
        for j in range(2**(N/2)):
            trun_state = Psi[dn*(2**N)+(j*2**(N/2)):dn*(2**N)+((j+1)*2**(N/2))]
            myA_Mat_new += np.outer(trun_state,np.conjugate(trun_state))
    return myA_Mat_new


def d_Mat_old(D,N, psi): #D is s N is L
    myd_Mat = np.zeros((D,D), dtype=complex)
    for x in range(D):
        for y in range(D):
            for i in  range(2**(N)):
                myd_Mat[x][y] = myd_Mat[x][y] + psi[x*(2**N)+i]*(np.conjugate(psi)[y*(2**N)+i])
    return myd_Mat

def d_Mat(D,N, Psi): #D is s N is 
    myd_Mat = np.zeros((D,D), dtype=complex)
    for x in range(D):
        for y in range(D):
            trun_stateX = Psi[x*2**(N):(x+1)*2**(N)]
            trun_stateY = Psi[y*2**(N):(y+1)*2**(N)]
            diags = np.multiply(trun_stateY,np.conjugate(trun_stateX))
            myd_Mat[x][y] = sum(diags)#sum(np.diagonal(np.outer(trun_stateY,np.conjugate(trun_stateX))))
    return myd_Mat

def Inf_fun(myA, myL, myd,  myPsi, Avg):
    S_list = list()
    N_list = list()
    #t0 = time.time()
    dMat = d_Mat(myd, myL, myPsi)
    #t1 = time.time()
    U_d = np.linalg.eigvals(dMat)
    #t2 = time.time()
    S_d = -sum(U_d*np.log(U_d+1e-30))
    #t3 = time.time()
    #print 'dmat:', t1-t0
    #print 'diag:', t2-t1
    #print 'Entan:',t3-t2
    I_l = 0
    EN = []
    avgloop = range(myL)
    if Avg == 'No':
        avgloop=np.array([0,myL/2])
    for l in avgloop:
        Roted = Rotate_fun(myA,l)
        Roted_d = np.tile(Roted, myd)
        Psi_rot=np.zeros((myd*2**myL), dtype=complex)
        for j in range(myd):
            Psi_rot[j*2**myL:(j+1)*(2**myL)] = [i for _,i in sorted(zip(Roted_d,myPsi[j*2**myL:(j+1)*(2**myL)]))]
        RhoA = Rho_A(myL, myd, Psi_rot)
        #A_Mat_old = A_Mat_fun(myL, Roted, myd, myPsi)
        U_A = np.linalg.eigvals(RhoA)
        S_A = -sum(U_A*np.log(U_A+1e-30))
        EN.append(sum(abs(U_A)-U_A)/2)
        S_list.append(S_A)
    #t4 = time.time()
    #print 'L loop:', t4-t3
    EN.append(sum(abs(U_d)-U_d)/2)
    S_list.append(S_d)
    return S_list, EN

#############################################3
## This part is for odd L

def Rho_A_odd_up(N, myd, Psi):
    myA_Mat_new = np.zeros((2**((N+1)/2),2**((N+1)/2)), dtype=complex)
    for dn in range(myd):
        for j in range(2**((N-1)/2)):
            trun_state = Psi[dn*(2**N)+(j*2**((N+1)/2)):dn*(2**N)+((j+1)*2**((N+1)/2))]
            myA_Mat_new += np.outer(trun_state,np.conjugate(trun_state))
    return myA_Mat_new


def Rho_A_odd_lw(N, myd, Psi):
    myA_Mat_new = np.zeros((2**((N-1)/2),2**((N-1)/2)), dtype=complex)
    for dn in range(myd):
        for j in range(2**((N+1)/2)):
            #print ' len Psi is:', Psi.shape, dn*(2**N)+(j*2**((N-1)/2)), dn*(2**N)+((j+1)*2**((N-1)/2))
            trun_state = Psi[dn*(2**N)+(j*2**((N-1)/2)):dn*(2**N)+((j+1)*2**((N-1)/2))]
            myA_Mat_new += np.outer(trun_state,np.conjugate(trun_state))
    return myA_Mat_new

def Inf_fun_odd(myA, myL, myd,  myPsi, Avg):
    S_list = list()
    N_list = list()
    S_list_up = list()
    S_list_lw = list()
    dMat = d_Mat(myd, myL, myPsi)
    U_d = np.linalg.eigvals(dMat)
    S_d = -sum(U_d*np.log(U_d+1e-30))
    EN_list = []
    avgloop = range(myL)
    if Avg == 'No':
        avgloop=np.array([0,(myL-1)/2])
    for l in avgloop:
        print 'l is>>>>', l
        Roted = Rotate_fun(myA,l)
        Roted_d = np.tile(Roted, myd)
        Psi_rot=np.zeros((myd*2**myL), dtype=complex)
        Psi_rot=np.zeros((myd*2**myL), dtype=complex)
        for j in range(myd):
            Psi_rot[j*2**myL:(j+1)*(2**myL)] = [i for _,i in sorted(zip(Roted_d,myPsi[j*2**myL:(j+1)*(2**myL)]))]
        RhoA_up = Rho_A_odd_up(myL, myd, Psi_rot)
        RhoA_lw = Rho_A_odd_lw(myL, myd, Psi_rot)
        #A_Mat_old = A_Mat_fun(myL, Roted, myd, myPsi)
        U_A_up = np.linalg.eigvals(RhoA_up)
        S_A_up = -sum(U_A_up*np.log(U_A_up+1e-30))
        U_A_lw = np.linalg.eigvals(RhoA_lw)
        S_A_lw = -sum(U_A_lw*np.log(U_A_lw+1e-30))
        EN_list.append(sum(abs(U_A_up)-U_A_up)/2)
        EN_list.append(sum(abs(U_A_lw)-U_A_lw)/2)
        #for eg in U_A_up:
        #    N_A = N_A + (abs(eg)-eg*1.0)/2.0
        #print '>>>', N_A
        #N_list.append(N_A)
        S_list_up.append(S_A_up)
        S_list_lw.append(S_A_lw)
    EN = list(EN_list[::2])+list(EN_list[1::2])
    EN.append(sum(abs(U_d)-U_d)/2)
    S_list.append(S_d)
    S_list_lw = np.roll(S_list_lw,int(myL/2))
    I_rVLS = np.array(list(S_list_lw)+ list(S_list_up)+[S_d])
    return I_rVLS, np.array(EN)
