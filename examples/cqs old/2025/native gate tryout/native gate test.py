from hardware.ionq.transpiler import *

theta = Parameter('theta')
q = QuantumRegister(2, "q")
qc = QuantumCircuit(q)
qc.append(IonQGPIGate(theta), [q[0]], [])
qc.append(IonQGPI2Gate(theta), [q[1]], [])
qc.append(IonQFullMSGate(), [q[0], q[1]])
qc.append(IonQPartialMSGate(theta), [q[0], q[1]], [])
qc.append(IonQVirtualZGate(theta), [q[0]], [])
qc.append(IonQZZGate(theta), [q[0], q[1]], [])
print(qc)



