# Select the Hardware
from hardware.eigens_acc.eigens_access import Hadamard_test as Hadamard_test_eigens
from hardware.qiskit_acc.qiskit_noiseless import Hadamard_test as Hadamard_test_qiskit_noiseless
from hardware.qiskit_acc.qiskit_noisy import Hadamard_test as Hadamard_test_qiskit_noisy

# from hardware.Qibo.qibo_access import Hadamard_test as Hadamard_test_qibo
# from hardware.Qibo.noisy_access import Hadamard_test as Hadamard_test_qibo_noisy
# from hardware.IonQ.ionq_access import Hadamard_test as Hadamard_test_ionq
# from hardware.IBMQ.ibmq_access import Hadamard_test as Hadamard_test_ibmq
# from hardware.AWS.braket_access import Hadamard_test as Hadamard_test_braket
# from hardware.AWS.noisy_access import Hadamard_test as Hadamard_test_braket_noisy


def Hadamard_test(n, U1, U2, Ub, real='r', backend='eigens', shots=1024, noise_level=None, device='SV1'):
    if backend == 'eigens':
        return Hadamard_test_eigens(n, U1, U2, Ub)
    elif backend == 'qiskit-noiseless':
        return Hadamard_test_qiskit_noiseless(n, U1, U2, Ub, shots)
    elif backend == 'qiskit-noisy':
        return Hadamard_test_qiskit_noisy(n, U1, U2, Ub, shots, noise_level)

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

    #
    # elif backend == 'braket_noisy':
    #     return Hadamard_test_braket_noisy(U, real, shots)
    else:
        raise ValueError("The backend should be in ['eigens', 'qiskit-noiseless', "
                         "'qiskit-noisy', 'ibmq', 'qiskit', 'braket', 'braket_noisy']")



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