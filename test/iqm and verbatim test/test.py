from transpiler.IQM_transpiler import transpile_to_IQM_braket
from qiskit.circuit.random import random_circuit
from braket.aws import AwsDevice

n = 5
d = 3
qc = random_circuit(n, d)
qc_iqm = transpile_to_IQM_braket(qc)
print(qc_iqm)

# Try with IQM
device = AwsDevice("arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet")

# Choose S3 bucket to store results
bucket = 'amazon-braket-prgroup-xiufan'# <<Update with your actual bucket name>> #eg: "amazon-braket-unique-aabbcdd"
prefix = "results"
s3_folder = (bucket, prefix)

task = device.run(qc_iqm, shots=2, disable_qubit_rewiring=True)
result = task.result()
print("Measurement Results:", result.measurement_counts)














