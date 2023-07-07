# 
# This code is part of Qton.
#
# 'qton_mpi_torch' is a version for MPI and PyTorch platform.
# 
# Qton Version: 1.0.0
# 
# This file: qton_mpi_torch\operators\_basic_op_.py
# Author(s): Yunheng Ma
# Timestamp: 2023-05-30 18:27:44
# 


__all__ = ["_Basic_op_"]


class _Basic_op_:
    '''
    Basic of operation on qubits.

    -Attributes(7):
        1. category --- is this a "gate", "channel", "superop" or else.
            type: str
        2. basename --- indicate this object is build from whom, e.g., 
            "Hadamard".
            type: str
        3. represent --- object representation.
            type: [list,] numpy.ndarray, 2D, complex
        4. num_ctrls --- number of controls.
            type: int
        5. num_qubits --- number of qubits.
            type: int
        6. num_params --- number of parameters.
            type: int
        7. dagger --- if dagger operation applied or not.
            type: bool
    '''
    category = ''
    basename = ''
    represent = None
    num_ctrls = 0
    num_qubits = 0
    num_params = 0
    dagger = False