# Select the Hardware
from hardware.Qibo.qibo_access import Hadamard_test as Hadamard_test_qibo
from hardware.IonQ.ionq_access import Hadamard_test as Hadamard_test_ionq
from hardware.Eigens.eigens_access import Hadamard_test as Hadamard_test_eigens
from hardware.IBMQ.ibmq_access import Hadamard_test as Hadamard_test_ibmq
from hardware.Matrix.matrix_access import Hadmard_test as Hadmard_test_matrix

def Hadamard_test(U, backend='qibo', alpha=1, shots=1024):
    if backend == 'qibo':
        expe = Hadamard_test_qibo(U, alpha, shots)
        return expe
    elif backend == 'ionq':
        return Hadamard_test_ionq(U, alpha, shots)
    elif backend == 'eigens':
        return Hadamard_test_eigens(U, alpha)
    elif backend == 'ibmq':
        return Hadamard_test_ibmq(U, alpha, shots)
    elif backend == 'matrix':
        return Hadmard_test_matrix(U, alpha)
    else:
        raise ValueError("The backend should be in ['qibo', 'ionq', 'eigens', 'ibmq', 'matrix']")