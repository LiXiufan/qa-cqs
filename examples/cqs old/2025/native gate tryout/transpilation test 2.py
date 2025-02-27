from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter
from hardware.ionq.transpiler import *
from numpy import pi
# qc = QuantumCircuit(3)
# qc.cx(0, 2)
# qc.x(0)
# qc.cz(0, 1)
# qc.cx(0, 2)
# qc.x(0)
# print(qc)
n = 2
# d = 4
# circ = random_circuit(n, d, measure=True)
theta = Parameter('theta')
theta = pi/2
q = QuantumRegister(n, "q")
qc = QuantumCircuit(q)
qc.append(IonQGPIGate(theta), [q[0]], [])
qc.append(IonQGPI2Gate(theta), [q[1]], [])
qc.append(IonQFullMSGate(), [q[0], q[1]])
qc.append(IonQPartialMSGate(theta), [q[0], q[1]], [])
qc.append(IonQVirtualZGate(theta), [q[0]], [])
qc.append(IonQZZGate(theta), [q[0], q[1]], [])

print("Pre-Transpilation: \n")
print(qc)
print(dict(qc.count_ops()))
print()
anc = QuantumRegister(1, 'ancilla')
qr = QuantumRegister(n, 'q')
cr = ClassicalRegister(n+1, 'c')
qc_ctrl = QuantumCircuit(anc, qr, cr)
qc_ctrl.append(qc.to_gate().control(ctrl_state='0'), [anc[0], *qr])

print("Controlled native gate circuit: \n")
print(qc_ctrl)
print()

transpiled_circ = transpile_to_ionq_native_gates(qc_ctrl)
print("Post-Transpilation: \n")
print(transpiled_circ)
print(dict(transpiled_circ.count_ops()))




# In total, there are 64 two qubit 'IonQ fully entangled MS gate'.