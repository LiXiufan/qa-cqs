from qiskit_ibm_runtime import QiskitRuntimeService

# Save an IBM Quantum account.

# QiskitRuntimeService.save_account(channel="ibm_quantum", token="MY_IBM_QUANTUM_TOKEN1")

service = QiskitRuntimeService()
program_inputs = {'iterations': 1}
options = {"backend_name": "ibmq_qasm_simulator"}
job = service.run(program_id="hello-world",
                options=options,
                inputs=program_inputs
                )
print(f"job id: {job.job_id()}")
result = job.result()
print(result)