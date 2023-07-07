# 
# This code is part of Qton.
#
# 'qton_mpi_torch' is a version for MPI and PyTorch platform.
# 
# Qton Version: 1.0.0
# 
# This file: qton_mpi_torch\__init__.py
# Author(s): Yunheng Ma
# Timestamp: 2023-05-30 18:27:44
# 


__version__ = '1.0.0'
__author__ = 'Yunheng Ma'


__all__ = ["Qcircuit"]


from .simulators.statevector import Qstatevector


def Qcircuit(total_num_qubits):
    '''
    Create a new quantum circuit instance.
    
    -In(1):
        1. total_num_qubits --- total number of qubits in the circuit.
            type: int
    
    -Return(1):
        1. --- quantum circuit instance.
            type: qton circuit instance.
    '''
    return Qstatevector(total_num_qubits)