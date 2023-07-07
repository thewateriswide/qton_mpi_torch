# 
# This code is part of Qton.
#
# 'qton_mpi_torch' is a version for MPI and PyTorch platform.
# 
# Qton Version: 1.0.0
# 
# This file: qton_mpi_torch\simulators\statevector.py
# Author(s): Yunheng Ma
# Timestamp: 2023-05-30 18:27:44
# 


__all__ = ["Qstatevector"]


from mpi4py import MPI
import numpy as np
import torch


# MPI world communicator.
MPI_COMM = MPI.COMM_WORLD
# MPI process rank.
MPI_RANK = MPI_COMM.Get_rank()
# MPI world communicator size.
MPI_SIZE = MPI_COMM.Get_size()

if torch.cuda.is_available() and True :  # False can forcibly disable the GPU.
    # Number of CUDA devices per node.
    CUDA_COUNT = torch.cuda.device_count()
    # CUDA device number corresponding to the current MPI process.
    CUDA_INDEX = MPI_RANK % CUDA_COUNT
    # The CUDA device currently being used by the process.
    DEVICE = torch.device(CUDA_INDEX)
else:
    DEVICE = torch.device('cpu')

# Storage and computational precision of state vectors.
TORCH_DTYPE = torch.complex128
NUMPY_DTYPE = np.complex128


# This is used to perform exchange operations between two local qubits.
Swap_tensor = torch.tensor([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0,1.],
    ], device=DEVICE, dtype=TORCH_DTYPE)


# Lower bound on the number of local bits per MPI process.
PROCESS_QUBIT_LIMIT = 2

# Alphabet for tensor contraction.
ALP = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 
    'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      ]


# Define a function to output error messages.
def error_print(words):
    '''
    Process 0 prints an error message and then terminates the entire program.
    
    - In(1):
        1. words --- string to print.
            type: str
    '''
    import inspect
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("ERROR [ %s ] : " % inspect.stack()[1][3] + words)
    exit()


from qton_mpi_torch.operators.gates import *


class Qstatevector(object):
    '''
    Quantum circuit represented by circuit statevector, running on MPI nodes.

    -Attributes(4):
        1. backend --- a simple tag for the simulator.
            type: str
        2. total_num_qubits --- total number of qubits in the circuit.
            type: int
        3. num_qubits --- number of local qubits.
            type: int
        4. state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
    
    -Methods(45):
        1.  __init__(self, total_num_qubits)
        2.  state2file(self, filename='.')
        3.  file2state(self, filename)
        4.  _apply_locally_(self, op, *qubits)
        5.  _temp_swap_(self, qubit)
        6.  apply_1q(self, rep, qubit)
        7.  apply_2q(self, rep, qubit1, qubit2)
        8.  apply(self, op, *qubits)
        9.  measure(self, qubit)
        10. sample(self, shots=1024, output='memory')
        11. i(self, qubit)
        12. x(self, qubit)
        13. y(self, qubit)
        14. z(self, qubit)
        15. h(self, qubit)
        16. s(self, qubit)
        17. t(self, qubit)
        18. sdg(self, qubit)
        19. tdg(self, qubit)
        20. rx(self, theta, qubit)
        21. ry(self, theta, qubit)
        22. rz(self, theta, qubit)
        23. p(self, phi, qubit)
        24. u1(self, lamda, qubit)
        25. u2(self, phi, lamda, qubit)
        26. u3(self, theta, phi, lamda, qubit)
        27. u(self, theta, phi, lamda, gamma, qubit)
        28. swap(self, qubit1, qubit2)
        29. cx(self, qubit1, qubit2)
        30. cy(self, qubit1, qubit2)
        31. cz(self, qubit1, qubit2)
        32. ch(self, qubit1, qubit2)
        33. cs(self, qubit1, qubit2)
        34. ct(self, qubit1, qubit2)
        35. csdg(self, qubit1, qubit2)
        36. ctdg(self, qubit1, qubit2)
        37. crx(self, theta, qubit1, qubit2)
        38. cry(self, theta, qubit1, qubit2)
        39. crz(self, theta, qubit1, qubit2)
        40. cp(self, phi, qubit1, qubit2)
        41. fsim(self, theta, phi, qubit1, qubit2)
        42. cu1(self, lamda, qubit1, qubit2)
        43. cu2(self, phi, lamda, qubit1, qubit2)
        44. cu3(self, theta, phi, lamda, qubit1, qubit2)
        45. cu(self, theta, phi, lamda, gamma, qubit1, qubit2)
    '''
    backend = 'statevector_mpi_torch'
    total_num_qubits = 0
    num_qubits = 0
    state = None


    def __init__(self, total_num_qubits):
        '''
        Check MPI runtime environment, allocate memory and set runtime
        parameters.

        -In(1):
            1. total_num_qubits --- total number of qubits in the circuit.
                type: int
        
        -Influenced(3):
            1. self.total_num_qubits --- total number of qubits in the circuit.
                type: int
            2. self.num_qubits --- number of local qubits.
                type: int
            3. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        # Check if the number of MPI processes launched is appropriate.
        Quotient, Remainder = divmod(2**total_num_qubits, MPI_SIZE)
        if Remainder != 0:
            error_print('The number of MPI processes cannot evenly divide the ' + 
                        'length of the quantum state.')
        if Quotient < 2**PROCESS_QUBIT_LIMIT:
            error_print('The length of the quantum state assigned to each ' + 
                        'process is too short to start the computation. Please ' + 
                        'reduce the number of MPI processes.')

        self.total_num_qubits = total_num_qubits
        self.num_qubits = int(np.log2(Quotient))
        self.state = torch.zeros(2**self.num_qubits, device=DEVICE, dtype=TORCH_DTYPE)

        if MPI_RANK == 0:
            self.state[0] = 1.0

        return None


    def state2file(self, filename='.'):
        '''
        Save the quantum state to a file.

        Note that the final quantum state may be too large and the write 
        operation may fail if there is not enough disk space available.

        - In(1):
            1. filename --- the file name.
                type: str
        
        - Return(1):
            1. filename --- the file name.
                type: str

        - Output(1):
            1. --- the statevector file.
                type: binary file
        '''
        import os
        import uuid

        # If no file name is specified, a random UUID will be generated as the 
        # file name.
        if filename == '.':
            if MPI_RANK == 0:
                while os.path.exists(filename):
                    filename = 'qstatevector_' + uuid.uuid4().hex
            filename = MPI_COMM.bcast(filename, root=0)

        buffer = np.empty(2**self.num_qubits, dtype=NUMPY_DTYPE)
        buffer = self.state.to('cpu').numpy()

        amode = MPI.MODE_WRONLY | MPI.MODE_CREATE
        fh = MPI.File.Open(MPI_COMM, filename, amode)
        offset = MPI_RANK * buffer.nbytes
        fh.Write_at_all(offset, buffer)
        fh.Close()
        del buffer

        return filename


    def file2state(self, filename):
        '''
        Read the quantum state from a file to the GPU memory.

        - In(1):
            1. filename --- the file name.
                type: str

        - Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        import os
        if os.path.exists(filename) == False:
            error_print('File not found.')

        filesize = os.path.getsize(filename)
        Quotient, Remainder = divmod(filesize, np.nbytes[NUMPY_DTYPE])
        if Remainder != 0 or np.log2(Quotient) % 1 != 0:
            error_print('Not a valid file.')

        if self.total_num_qubits != int(np.log2(Quotient)):
            error_print('Mismatch with the current number of qubits.')

        buffer = np.empty(2**self.num_qubits, dtype=NUMPY_DTYPE)

        amode = MPI.MODE_RDONLY
        fh = MPI.File.Open(MPI_COMM, filename, amode)
        offset = MPI_RANK * buffer.nbytes
        fh.Read_at_all(offset, buffer)
        fh.Close()

        self.state = torch.from_numpy(buffer).to(DEVICE)
        del buffer

        return None


    def _apply_locally_(self, rep, *qubits):
        '''
        Each process independently executes a gate operation, only applicable 
        when the qubit is a local qubit.

        -In(2):
            1. op --- gate operation to be applied.
                type: numpy.ndarray
            2. qubits --- qubit(s) to be acted upon.
                type: int; list, int
        
        -Influenced(1):
            1. self.state ---- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        nq_gate = int(np.log2(rep.shape[0]))
        a_idx = [*range(nq_gate, 2*nq_gate)]
        b_idx = [self.num_qubits-i-1 for i in qubits]
        rep = rep.reshape([2]*2*nq_gate)
        self.state = self.state.reshape([2]*self.num_qubits)

        # tensor contraction operation.
        self.state = torch.tensordot(rep, self.state, dims=(a_idx, b_idx))

        # Rearrange tensor indices to the correct order.
        s = ''.join(ALP[:self.num_qubits])
        end = s
        start = ''
        for i in range(nq_gate):
            start += end[self.num_qubits-qubits[i]-1]
            s = s.replace(start[i], '')
        start = start + s
        self.state = torch.einsum(start+'->'+end, self.state).reshape(-1)

        torch.cuda.empty_cache()  # Free up unreferenced variables' GPU memory.
        return None


    def _temp_swap_(self, qubit):
        '''
        Swap the given global qubit with the highest local qubit, which equals
        to swaping the second half of quantum state of processes Pi's with the
        first half of quantum state of processes Pj's. Pi's and Pj's are MPI
        processes that have values 0 and 1 respectively on the given global
        qubit.

        For a faster point-to-point GPU communication, one can rewrite this 
        function using the 'torch.distributed' module based on the target
        platform.

        -In(1):
            1. qubit --- the given global qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        half = 2**self.num_qubits // 2
        stride = 2**(qubit - self.num_qubits)  

        # send & receive buffers on CPU
        sendbuff = np.empty(half, dtype=NUMPY_DTYPE)
        recvbuff = np.empty(half, dtype=NUMPY_DTYPE)

        if MPI_RANK & stride:  # MPI ranks of Pj's, send first half
            MPI_COMM.Recv(recvbuff, source=MPI_RANK-stride, tag=100)
            sendbuff = self.state[:half].to('cpu').numpy()
            MPI_COMM.Send(sendbuff, dest=MPI_RANK-stride, tag=101)
            self.state[:half] = torch.from_numpy(recvbuff).to(DEVICE)
        else:  # MPI ranks of Pi's, send second half
            sendbuff = self.state[half:].to('cpu').numpy()
            MPI_COMM.Send(sendbuff, dest=MPI_RANK+stride, tag=100)
            MPI_COMM.Recv(recvbuff, source=MPI_RANK+stride, tag=101)
            self.state[half:] = torch.from_numpy(recvbuff).to(DEVICE)
        del sendbuff, recvbuff

        return None


    def apply_1q(self, rep, qubit):
        '''
        Apply a single-qubit gate. If the given qubit is a global qubit, it
        requires swap operations between the global and local qubits to be
        performed.

        -In(2):
            1. rep --- gate operation to be applied.
                type: torch.Tensor
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        if qubit >= self.total_num_qubits:
            error_print('Qubit index exceeds maximum allowable value.')

        if qubit < self.num_qubits:
            self._apply_locally_(rep, qubit)
        else:
            self._temp_swap_(qubit)
            self._apply_locally_(rep, self.num_qubits-1) 
            self._temp_swap_(qubit)

        return None


    def apply_2q(self, rep, qubit1, qubit2):
        '''
        Apply a double-qubit gate, applying global and local swap operations
        based on different situations.

        -In(3):
            1. rep --- gate operation to be applied.
                type: torch.Tensor
            2. qubit1 --- first qubit, usually considered as a control qubit.
                type: int
            3. qubit2 --- second qubit, usually considered as a target qubit.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        if qubit1 >= self.total_num_qubits or qubit2 >= self.total_num_qubits:
            error_print('At least one qubit index exceeds maximum allowable ' +
                        'value.')

        if qubit1 == qubit2:
            error_print('The indices of two qubits cannot be the same.')

        # |global_qubits: ...>|local_qubits: L1,L2,...>
        L1 = self.num_qubits - 1
        L2 = self.num_qubits - 2

        if qubit2 < self.num_qubits:  # qubit2 is local

            if qubit1 < self.num_qubits:  # qubit1 is local
                self._apply_locally_(rep, qubit1, qubit2)

            else:  # qubit1 is global
                if qubit2 != L1:  # landing spot is clear
                    self._temp_swap_(qubit1)
                    self._apply_locally_(rep, L1, qubit2)
                    self._temp_swap_(qubit1)

                else:  # qubit2 occupies landing spot
                    self._apply_locally_(Swap_tensor, qubit2, L2)  # swap qubit2 to next
                    self._temp_swap_(qubit1)
                    self._apply_locally_(rep, L1, L2)
                    self._temp_swap_(qubit1)
                    self._apply_locally_(Swap_tensor, qubit2, L2)

        else:  # qubit2 is global

            if qubit1 < self.num_qubits:  # qubit1 is local

                if qubit1 != L1:  # landing spot is clear
                    self._temp_swap_(qubit2)
                    self._apply_locally_(rep, qubit1, L1)
                    self._temp_swap_(qubit2)

                else:  # qubit1 occupies landing spot
                    self._apply_locally_(Swap_tensor, qubit1, L2)
                    self._temp_swap_(qubit2)
                    self._apply_locally_(rep, L2, L1)
                    self._temp_swap_(qubit2)
                    self._apply_locally_(Swap_tensor, qubit1, L2)

            else:  # qubit1 is global
                self._temp_swap_(qubit2)  # qubit2 lands firstly
                self._apply_locally_(Swap_tensor, L1, L2)
                self._temp_swap_(qubit1)
                self._apply_locally_(rep, L1, L2)
                self._temp_swap_(qubit1)
                self._apply_locally_(Swap_tensor, L1, L2)
                self._temp_swap_(qubit2)

        return None
    

    def apply(self, op, *qubits):
        '''
        Apply a single-qubit gate or double-qubit gate, depending on the number
        of qubits given and the width of the gate.

        This function integrates 'self.apply_1q' and 'self.apply_2q'.

        -In(2):
            1. op --- gate operation to be applied.
                type: qton gate
            2. qubits --- qubit(s) to be acted upon.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
        '''
        nq = len(qubits)
        if nq == 0:
            error_print('Missing qubits to be acted upon.')

        for i in range(nq):
            if type(qubits[i]) is not int:
                error_print('Invalid bit parameter, only accepts integer values.')
        
        if op.num_qubits > 2 or  nq > 2:
            error_print('Gate operations with a width greater than two qubits ' + 
                        'are currently not supported.')
            
        if op.num_qubits != nq:
            error_print('The width of the quantum gate does not match the ' + 
                        'number of qubits to be operated on.')

        if nq == 1:
            rep = torch.from_numpy(op.represent).to(DEVICE)
            rep = rep.type(TORCH_DTYPE)
            self.apply_1q(rep, qubits[0])
            MPI_COMM.barrier()

        elif nq == 2:
            if len(set(qubits)) < 2:
                error_print('Cannot act on two identical qubits.')
            rep = torch.from_numpy(op.represent).to(DEVICE)
            rep = rep.type(TORCH_DTYPE)
            self.apply_2q(rep, qubits[0], qubits[1])
            MPI_COMM.barrier()

        return None


    def measure(self, qubit):
        '''
        Perform a projective measurement on the given qubit.

        In this version, a measured qubit cannot be deleted.

        For clarity, only one qubit is allowed to be measured at a time.
        
        - In(1):
            1. qubit --- the qubit to be measured.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex
                
        - Return(1):
            1. bit --- output of the measurement, 0 or 1.
                type: int
        '''
        if qubit < self.num_qubits:  # if a local qubit
            p1 = 0.
            for idx in range(2**self.num_qubits):
                if idx & 2**qubit:  # if |qubit> = |1>
                    p1 += abs(self.state[idx].item())**2

            probability1 = MPI_COMM.reduce(p1, MPI.SUM, root=0)
            probability1 = MPI_COMM.bcast(probability1, root=0)

            bit = None
            if MPI_RANK == 0:
                bit = int(np.random.random() < probability1)
            bit = MPI_COMM.bcast(bit, root=0)

            if bit:
                for idx in range(2**self.num_qubits):
                    if idx & 2**qubit:
                        self.state[idx] /= np.sqrt(probability1)
                    else:
                        self.state[idx] = 0.
            else:
                for idx in range(2**self.num_qubits):
                    if idx & 2**qubit:
                        self.state[idx] = 0.
                    else:
                        self.state[idx] /= np.sqrt(1. - probability1)

        else:  # if a global qubit.
            p1 = 0.
            if MPI_RANK & 2**(qubit - self.num_qubits):  # if |qubit> = |1>
                p1 = (self.state * self.state.conj()).sum().item().real
            
            probability1 = MPI_COMM.reduce(p1, MPI.SUM, root=0)
            probability1 = MPI_COMM.bcast(probability1, root=0)

            bit = None
            if MPI_RANK == 0:
                bit = int(np.random.random() < probability1)
            bit = MPI_COMM.bcast(bit, root=0)

            if bit:
                if MPI_RANK & 2**(qubit - self.num_qubits):
                    self.state[:] /= np.sqrt(probability1)
                else:
                    self.state[:] = 0.
            else:
                if MPI_RANK & 2**(qubit - self.num_qubits):
                    self.state[:] = 0.
                else:
                    self.state[:] /= np.sqrt(1. - probability1)

        return bit


    def sample(self, shots=1024, output='memory'):
        '''
        Sample the quantum state. This procedure does not affect the data in 
        the GPU memory.
        
        - In(1):
            1. shots --- the total number of samples.
                type: int
            2. output --- format for outputing sampling results.
                type: str: "memory", "statistic", "counts"

        - Return(1):
            1.(3):
                1. memory --- sampled temporal results.
                    type: list, int
                2. statistic --- statistical analysis of sampled results for 
                    each outcome.
                    type: numpy.ndarray, int
                3. counts --- counts for each outcome.
                    type: dict        
        '''
        from random import choices

        distribution = np.empty(2**self.num_qubits, dtype=np.float64)
        distribution = abs((self.state * self.state.conj()).to('cpu').numpy())

        my_weight = distribution.sum() * shots
        weight_tab = MPI_COMM.gather(my_weight, root=0)

        shots_tab = None
        if MPI_RANK == 0:
            shots_tab = np.array(weight_tab, dtype=int)
            remain = shots - shots_tab.sum()
            temp = choices(range(MPI_SIZE), weight_tab, k=remain)  # allocate the remaining shots.
            for idx in temp:
                shots_tab[idx] += 1

        # shots assigned to every process.
        my_shots = MPI_COMM.scatter(shots_tab, root=0)

        if my_shots > 0:
            memory = choices(range(2**self.num_qubits), distribution, k=my_shots)
        else:
            memory = []
        
        # Attention, all outputs below correspond only to local qubits.
        if output == 'memory':
            return memory

        elif output == 'statistic':
            statistic = np.zeros(2**self.num_qubits, int)
            for i in memory:
                statistic[i] += 1
            return statistic

        elif output == 'counts':
            counts = {}
            for i in memory:
                key = format(i, '0%db' % self.num_qubits)
                if key in counts:
                    counts[key] += 1
                else:
                    counts[key] = 1
            return counts

        else:
            error_print('Unrecognized output type.')
        
# 
# Gate methods.
# 

    def i(self, qubit):
        '''
        Identity gate.
        
        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(I_gate(), qubit)
        return None


    def x(self, qubit):
        '''
        Pauli-X gate.
        
        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(X_gate(), qubit)
        return None


    def y(self, qubit):
        '''
        Pauli-Y gate.
        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Y_gate(), qubit)
        return None


    def z(self, qubit):
        '''
        Pauli-Z gate.

        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Z_gate(), qubit)
        return None


    def h(self, qubit):
        '''
        Hadamard gate.

        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(H_gate(), qubit)
        return None


    def s(self, qubit):
        '''
        Phase S gate.

        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(S_gate(), qubit)
        return None


    def t(self, qubit):
        '''
        pi/8 T gate.

        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(T_gate(), qubit)
        return None


    def sdg(self, qubit):
        '''
        S dagger gate.
        
        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(S_gate(dagger=True), qubit)
        return None


    def tdg(self, qubit):
        '''
        T dagger gate.
        
        -In(1):
            1. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(T_gate(dagger=True), qubit)
        return None


    def rx(self, theta, qubit):
        '''
        Rotation along X axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Rx_gate([theta]), qubit)
        return None


    def ry(self, theta, qubit):
        '''
        Rotation along Y axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Ry_gate([theta]), qubit)
        return None


    def rz(self, theta, qubit):
        '''
        Rotation along Z axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Rz_gate([theta]), qubit)
        return None


    def p(self, phi, qubit):
        '''
        Phase gate.

        -In(2):
            1. phi --- phase angle.
                type: float
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(P_gate([phi]), qubit)
        return None


    def u1(self, lamda, qubit):
        '''
        U1 gate.
        
        -In(2):
            1. lamda --- phase angle.
                type: float
            2. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U1_gate([lamda]), qubit)
        return None


    def u2(self, phi, lamda, qubit):
        '''
        U2 gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U2_gate([phi, lamda]), qubit)
        return None


    def u3(self, theta, phi, lamda, qubit):
        '''
        U3 gate.

        -In(4):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U3_gate([theta, phi, lamda]), qubit)
        return None


    def u(self, theta, phi, lamda, gamma, qubit):
        '''
        Universal gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubit --- qubit to be acted upon.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U_gate([theta, phi, lamda, gamma]), qubit)
        return None


    def swap(self, qubit1, qubit2):
        '''
        Swap gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int
        
        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Swap_gate(), qubit1, qubit2)
        return None


    def cx(self, qubit1, qubit2):
        '''
        Controlled Pauli-X gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(X_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def cy(self, qubit1, qubit2):
        '''
        Controlled Pauli-Y gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Y_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def cz(self, qubit1, qubit2):
        '''
        Controlled Pauli-Z gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Z_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def ch(self, qubit1, qubit2):
        '''
        Controlled Hadamard gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(H_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def cs(self, qubit1, qubit2):
        '''
        Controlled S gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(S_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def ct(self, qubit1, qubit2):
        '''
        Controlled T gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(T_gate(num_ctrls=1), qubit1, qubit2)
        return None


    def csdg(self, qubit1, qubit2):
        '''
        Controlled S dagger gate.

        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(S_gate(num_ctrls=1, dagger=True), qubit1, qubit2)
        return None


    def ctdg(self, qubit1, qubit2):
        '''
        Controlled T dagger gate.
        
        -In(2):
            1. qubit1 --- first qubit.
                type: int
            2. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(T_gate(num_ctrls=1, dagger=True), qubit1, qubit2)
        return None


    def crx(self, theta, qubit1, qubit2):
        '''
        Controlled rotation along X axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubit1 --- first qubit.
                type: int
            3. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Rx_gate([theta], num_ctrls=1), qubit1, qubit2)
        return None


    def cry(self, theta, qubit1, qubit2):
        '''
        Controlled rotation along Y axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubit1 --- first qubit.
                type: int
            3. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Ry_gate([theta], num_ctrls=1), qubit1, qubit2)
        return None


    def crz(self, theta, qubit1, qubit2):
        '''
        Controlled rotation along Z axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubit1 --- first qubit.
                type: int
            3. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Rz_gate([theta], num_ctrls=1), qubit1, qubit2)
        return None


    def cp(self, phi, qubit1, qubit2):
        '''
        Controlled phase gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. qubit1 --- first qubit.
                type: int
            3. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(P_gate([phi], num_ctrls=1), qubit1, qubit2)
        return None


    def fsim(self, theta, phi, qubit1, qubit2):
        '''
        fSim gate.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. qubit1 --- first qubit.
                type: int
            4. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(Fsim_gate([theta, phi]), qubit1, qubit2)
        return None


    def cu1(self, lamda, qubit1, qubit2):
        '''
        Controlled U1 gate.

        -In(3):
            1. lamda --- phase angle.
                type: float
            2. qubit1 --- first qubit.
                type: int
            3. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U1_gate([lamda], num_ctrls=1), qubit1, qubit2)
        return None


    def cu2(self, phi, lamda, qubit1, qubit2):
        '''
        Controlled U2 gate.
        
        -In(4):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubit1 --- first qubit.
                type: int
            4. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U2_gate([phi, lamda], num_ctrls=1), qubit1, qubit2)
        return None


    def cu3(self, theta, phi, lamda, qubit1, qubit2):
        '''
        Controlled U3 gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubit1 --- first qubit.
                type: int
            5. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U3_gate([theta, phi, lamda], num_ctrls=1), qubit1, qubit2)
        return None


    def cu(self, theta, phi, lamda, gamma, qubit1, qubit2):
        '''
        Controlled universal gate.

        -In(6):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubit1 --- first qubit.
                type: int
            6. qubit2 --- second qubit.
                type: int

        -Influenced(1):
            1. self.state --- vector representation of the local quantum state.
                type: torch.Tensor, complex    
        '''
        self.apply(U_gate([theta, phi, lamda, gamma], num_ctrls=1), qubit1, qubit2)
        return None