# Select the Hardware
# from hardware.Qibo.qibo_access import Hadamard_test as Hadamard_test_qibo
# from hardware.Qibo.noisy_access import Hadamard_test as Hadamard_test_qibo_noisy
# from hardware.IonQ.ionq_access import Hadamard_test as Hadamard_test_ionq
from hardware.Eigens.eigens_access import Hadamard_test as Hadamard_test_eigens
# from hardware.IBMQ.ibmq_access import Hadamard_test as Hadamard_test_ibmq
# from hardware.Matrix.matrix_access import Hadmard_test as Hadmard_test_matrix
# from hardware.AWS.braket_access import Hadamard_test as Hadamard_test_braket
# from hardware.AWS.noisy_access import Hadamard_test as Hadamard_test_braket_noisy
def Hadamard_test(n, U1, U2, real='r', backend='eigens', shots=1024, device='SV1'):
    if backend == 'eigens':
        return Hadamard_test_eigens(n, U1, U2)
    # elif backend == 'braket':
    #     return Hadamard_test_braket(n, U1, U2, real=real, device=device, shots=shots)
    # elif backend == 'qibo':
    #     return Hadamard_test_qibo(U, real, shots)
    # elif backend == 'qibo_noisy':
    #     return Hadamard_test_qibo_noisy(U, real, shots)
    # elif backend == 'ionq':
    #     return Hadamard_test_ionq(U, real, shots)
    # elif backend == 'ibmq':
    #     return Hadamard_test_ibmq(U, real, shots)
    # elif backend == 'matrix':
    #     return Hadmard_test_matrix(U, real)
    #
    # elif backend == 'braket_noisy':
    #     return Hadamard_test_braket_noisy(U, real, shots)
    else:
        raise ValueError("The backend should be in ['qibo', 'qibo_noisy', "
                         "'ionq', 'eigens', 'ibmq', 'matrix', 'braket', 'braket_noisy']")



    # width = len(U[0])
    # depth = len(U)
    # counter = 0
    # for u in U:
    #     for g in u:
    #         if g == 'I':
    #             counter += 1
    # # If the circuit is composed of identities, we return 1 as the real part and return 0 as the imaginary part
    # if counter == width * depth:
    #     if alpha == 1:
    #         return 1
    #     else:
    #         return 0
    # else: