#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from itertools import cycle
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator
from scipy.linalg import expm
import csv
import sys 


# In[2]:


def swap_ind(state_r,ind): #auxiliary function swaps indices 
    for i in range (2**(ind-1)):
        state_r[[2*i,2*i+1],:]=state_r[[2*i+1,2*i],:]
    return state_r

def swap_sign(state_r,ind): #auxiliary function swaps signs
    for i in range (2**(ind-1)):
        state_r[2*i,:]=-state_r[2*i,:]
    return state_r

def x (state, ind): #X Pauli gate
    ind=ind+1
    state_c=np.copy(state)
    state_r=np.reshape(state_c,(2**ind,2**(n-ind)))
    state_r=swap_ind(state_r,ind)
    return state_r.flatten()

def y (state, ind): #Y Pauli gate
    ind=ind+1
    state_c=np.copy(state)
    state_r=np.reshape(state_c,(2**ind,2**(n-ind)))
    state_r=swap_ind(state_r,ind)
    state_r=swap_sign(state_r,ind)
    return 1j*state_r.flatten()


def z (state, ind): #Y Pauli gate
    ind=ind+1
    state_c=np.copy(state)
    state_r=np.reshape(state_c,(2**ind,2**(n-ind)))
    state_r=-swap_sign(state_r,ind)
    return state_r.flatten()

def rx (state, theta, ind):
    state_c=np.copy(state)
    state_X=x (state_c, ind)
    state_Rx=np.cos(theta)*state-1j*np.sin(theta)*state_X
    return state_Rx

def ry (state, theta, ind):
    state_c=np.copy(state)
    state_Y=y (state_c, ind)
    state_Ry=np.cos(theta)*state-1j*np.sin(theta)*state_Y
    return state_Ry

def rz (state, theta, ind):
    state_c=np.copy(state)
    state_Z=z (state_c, ind) 
    state_Rz=np.cos(theta)*state-1j*np.sin(theta)*state_Z
    return state_Rz


def ion1 (state, theta, ReE, ImE, Del, ind): #the hamiltonian is shifteed as we ignore the Identity term, Del= \omega-\omega_e+\omega_g, theta \in [0,T_max] such that |E| is constant 
    A=np.array([ReE,ImE,Del])
    a=np.linalg.norm(A)
    
    state_c=np.copy(state)
    state_X=x (state_c, ind)
    state_Y=y (state_c, ind)
    state_Z=z (state_c, ind) 
    state_ion1=np.cos(theta*a)*state_c-1j*np.sin(theta*a)*((ReE*state_X+ImE*state_Y+Del*state_Z)/a)
    return state_ion1
    


def rxx (state, theta, ind1, ind2):
    state_c=np.copy(state)
    state_X=x (state_c, ind1)
    state_X=x (state_X, ind2)
    state_Rx=np.cos(theta)*state-1j*np.sin(theta)*state_X
    return state_Rx


# In[3]:


# (rz o rz)(rxx)(rx o rx)
def ising_entangler (state,pc,ind1,ind2):
    state=rz(state,next(pc),ind1)
    state=rz(state,next(pc),ind2)
    state=rxx(state,next(pc),ind1,ind2)
    state=rx(state,next(pc),ind1)
    state=rx(state,next(pc),ind2)
    return state

    
def checkerboard (state,n,p,pc):
    for _ in range(p):
        for ind_1 in range(n//2):
            state=ising_entangler(state,pc,2*ind_1,2*ind_1+1)
        for ind_2 in range(n//2-1):
            state=ising_entangler(state,pc,2*ind_2+1,2*ind_2+2)
        state=ising_entangler(state,pc,0,n-1)
    return state

def ion_1qubit_ansatz(state,p,pc):
    for _ in range(p):
        state=ion1 (state, next(pc), next(pc), next(pc), next(pc), 0)
    return state

def der_ising_entangler (state,pc,ind1,ind2,j):
    
    state=rz(state,next(pc),ind1)
    state=rz(state,next(pc),ind2)
    state=rxx(state,next(pc),ind1,ind2)
    state=rx(state,next(pc),ind1)
    state=rx(state,next(pc),ind2)
    return state
    
def matrix_U (n,p,pc):
    zeros=np.zeros(2**n) #initial state
    matrix=[np.zeros(2**n)]
    for i in range(2**n):
        zeros[i]=1
        psi=checkerboard(zeros,n,p,pc)
        matrix=np.concatenate((matrix,[psi]),axis=0)
        zeros[i]=0
    return matrix[1:].T

def k_tofolli(psi): #limited to when this is applied to all qubits and controlling the last one. Works with matrices and vectors
    c=np.copy(psi[-1])
    psi[-1]=psi[-2]
    psi[-2]=c
    return psi


def rand_params(n,p):
    if n % 2 == 0:
        num_params=5*n*p
    else:
        num_params=5*(n-1)*p
    params=np.random.rand(num_params)*2*pi
    return params, num_params

def random_U(n):
    A=np.random.rand(2**n,2**n)+1j*np.random.rand(2**n,2**n)
    H=A.conj().T.dot(A)
    U=expm(1j*H*np.random.rand()*2*np.pi) #target Unitary
    return U


# ## USE THIS TO SAVE INTO CSV FILE

# In[4]:


def file_dump(line,name):
    with open(path+name,'a') as f: #####   ADD HERE PATH
        w=csv.writer(f,delimiter=',')
        w.writerow(line)


# ## Compilation of a random unitary

# In[10]:


path=""
file_dump([1],"soumik.csv")


# In[105]:



def compiler(n,p,t,goat=True):

   def cost (params):
       pc=cycle(params)#parmeters iterator
       U=matrix_U(n,p,pc)
       cost=1-(np.absolute(np.trace(U.dot(t.conj().T)))**2)/(2**(2*n))
       return cost

   def der_cost(params):
       dim2=(2**(2*n))
       pc=cycle(params)
       U=matrix_U(n,p,pc)
       tr=np.trace(U.dot(t.conj().T))
       grad=[]
       der_params=np.copy(params)
       for j in range(num_params):
           der_params[j]=der_params[j]+np.pi/2
           d_pc=cycle(der_params)
           d_U=matrix_U(n,p,d_pc)
           d_tr=np.trace(d_U.dot(t.conj().T))
           d_cost=-2*np.real(d_tr.conj()*tr)/dim2
           grad.append(d_cost)
           der_params[j]=der_params[j]-np.pi/2

       return grad

   params,num_params=rand_params(n,p)
   bds=[(0,2*np.pi)]*num_params
   
   t0=time.time()
   
   if goat==False:    
       res =  minimize(cost,params ,method='L-BFGS-B', jac='3-point',bounds=bds, options={'maxiter': 100} ) #THIS IS THE OPTIMIZER
   elif goat==True:
       res =  minimize(cost,params ,method='L-BFGS-B', jac=der_cost, bounds=bds,options={'maxiter': 100}) #THIS IS THE OPTIMIZER
   
   tf=time.time()-t0
   
   return res.fun, res.x, tf


# In[111]:


#    ODD ARE NUMERICAL EVEN ARE GOAT!!!!, each pair has the same seed
n=int(sys.argv[1])
p=int(sys.argv[2])
path=sys.argv[3]
ind=int(sys.argv[4])


seed=int(np.floor(ind/2))
np.random.seed(seed)

t=random_U(n)

if ind%2==0: #this is for goat
    mode='GOAT'
    cost, params, tot_time = compiler(n,p,t,goat=True)
else: #this is numerical
    mode='NUM'
    cost, params, tot_time = compiler(n,p,t,goat=False)
    
file_dump([cost],f'{n}n_{p}p_{seed}seed_{mode}.csv')
file_dump([tot_time],f'{n}n_{p}p_{seed}seed_{mode}.csv')
file_dump(params,f'{n}n_{p}p_{seed}seed_{mode}.csv')

