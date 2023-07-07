# 
# This code is part of Qton.
#
# 'qton_mpi_torch' is a version for MPI and PyTorch platform.
# 
# Qton Version: 1.0.0
# 
# This file: qton_mpi_torch\operators\__init__.py
# Author(s): Yunheng Ma
# Timestamp: 2023-05-30 18:27:44
# 


__all__ = ["H_gate",
           "I_gate",
           "X_gate",
           "Y_gate",
           "Z_gate",
           "S_gate",
           "T_gate",
           "Swap_gate",
           "P_gate",
           "U_gate",
           "Rx_gate",
           "Ry_gate",
           "Rz_gate",
           "U1_gate",
           "U2_gate",
           "U3_gate",
           "Fsim_gate",
        #    "I",
        #    "X",
        #    "Y",
        #    "Z",
        #    "H",
        #    "S",
        #    "T",
        #    "Swap",
        #    "CX",
        #    "CY",
        #    "CZ",
        #    "CH",
        #    "CS",
        #    "CT",
        #    "RX",
        #    "RY",
        #    "RZ",
        #    "P",
        #    "U1",
        #    "U2",
        #    "U3",
        #    "U",
        #    "FSIM",
        #    "CRX",
        #    "CRY",
        #    "CRZ",
        #    "CP",
        #    "CU1",
        #    "CU2",
        #    "CU3",
        #    "CU",
           ]


from .gates import *


# 
# 1-qubit fixed quantum gates.
# 

I = I_gate().represent
X = X_gate().represent
Y = Y_gate().represent
Z = Z_gate().represent
H = H_gate().represent
S = S_gate().represent
T = T_gate().represent


# 
# 2-qubit fixed quantum gates.
# 

Swap = Swap_gate().represent
CX = X_gate(num_ctrls=1).represent
CY = Y_gate(num_ctrls=1).represent
CZ = Z_gate(num_ctrls=1).represent
CH = H_gate(num_ctrls=1).represent
CS = S_gate(num_ctrls=1).represent
CT = T_gate(num_ctrls=1).represent


# 
# 1-qubit quantum gates, with parameters.
# 

def RX(theta): 
    return Rx_gate(params=[theta]).represent
def RY(theta): 
    return Ry_gate(params=[theta]).represent
def RZ(theta): 
    return Rz_gate(params=[theta]).represent

def P (phi):               
    return P_gate (params=[phi]).represent
def U1(lamda):             
    return U1_gate(params=[lamda]).represent
def U2(phi, lamda):        
    return U2_gate(params=[phi, lamda]).represent
def U3(theta, phi, lamda): 
    return U3_gate(params=[theta, phi, lamda]).represent
def U (theta, phi, lamda): 
    return U_gate (params=[theta, phi, lamda]).represent


# 
# 2-qubit quantum gates, with parameters.
# 

def FSIM(theta, phi):
    return Fsim_gate(params=[theta, phi]).represent

def CRX(theta): 
    return Rx_gate(params=[theta], num_ctrls=1).represent
def CRY(theta): 
    return Ry_gate(params=[theta], num_ctrls=1).represent
def CRZ(theta): 
    return Rz_gate(params=[theta], num_ctrls=1).represent

def CP (phi):               
    return P_gate (params=[phi], num_ctrls=1).represent
def CU1(lamda):             
    return U1_gate(params=[lamda], num_ctrls=1).represent
def CU2(phi, lamda):        
    return U2_gate(params=[phi, lamda], num_ctrls=1).represent
def CU3(theta, phi, lamda): 
    return U3_gate(params=[theta, phi, lamda], num_ctrls=1).represent
def CU (theta, phi, lamda): 
    return U_gate (params=[theta, phi, lamda], num_ctrls=1).represent