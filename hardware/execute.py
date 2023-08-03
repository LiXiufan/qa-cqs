# Select the Hardware
from hardware.Qibo.qibo_access import Hadamard_test as Hadamard_test_qibo
from hardware.Qibo.noisy_access import Hadamard_test as Hadamard_test_qibo_noisy
from hardware.IonQ.ionq_access import Hadamard_test as Hadamard_test_ionq
from hardware.Eigens.eigens_access import Hadamard_test as Hadamard_test_eigens
from hardware.IBMQ.ibmq_access import Hadamard_test as Hadamard_test_ibmq
from hardware.Matrix.matrix_access import Hadmard_test as Hadmard_test_matrix
from hardware.AWS.braket_access import Hadamard_test as Hadamard_test_braket
from hardware.AWS.noisy_access import Hadamard_test as Hadamard_test_braket_noisy

def Hadamard_test(U, backend='qibo', alpha=1, shots=1024, tasks_num = 0, shots_num = 0):
    width = len(U[0])
    depth = len(U)
    counter = 0
    for u in U:
        for g in u:
            if g == 'I':
                counter += 1
    # If the circuit is composed of identities, we return 1 as the real part and return 0 as the imaginary part
    if counter == width * depth:
        if alpha == 1:
            return 1, tasks_num, shots_num
        else:
            return 0, tasks_num, shots_num
    else:
        if backend == 'qibo':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_qibo(U, alpha, shots), tasks_num, shots_num
        elif backend == 'qibo_noisy':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_qibo_noisy(U, alpha, shots), tasks_num, shots_num
        elif backend == 'ionq':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_ionq(U, alpha, shots), tasks_num, shots_num
        elif backend == 'eigens':
            return Hadamard_test_eigens(U, alpha), tasks_num, shots_num
        elif backend == 'ibmq':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_ibmq(U, alpha, shots), tasks_num, shots_num
        elif backend == 'matrix':
            return Hadmard_test_matrix(U, alpha), tasks_num, shots_num
        elif backend == 'braket':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_braket(U, alpha, shots), tasks_num, shots_num
        elif backend == 'braket_noisy':
            tasks_num += 1
            shots_num += shots
            return Hadamard_test_braket_noisy(U, alpha, shots), tasks_num, shots_num
        else:
            raise ValueError("The backend should be in ['qibo', 'qibo_noisy', "
                             "'ionq', 'eigens', 'ibmq', 'matrix', 'braket', 'braket_noisy']")



