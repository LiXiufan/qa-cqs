# from qiskit_braket_provider import AWSBraketProvider
#
# provider = AWSBraketProvider()
# backend = provider.backends()
# print(backend)


import boto3

# Replace eu-west-2 with the region you created the quantum tasks in
# client = boto3.client('resourcegroupstaggingapi', region_name="us-east-1")

# client = boto3.client('resourcegroupstaggingapi')
# response = client.get_resources(
#     TagFilters=[
#         {
#             'Key': 'batch'
#         },
#     ],
# )
# tasks = [t["ResourceARN"] for t in response["ResourceTagMappingList"]]
# print(tasks)


braket = boto3.client("braket")
# response = braket.search_quantum_tasks(filters=[{
#     'name': 'arn:aws:braket:us-east-1::device/qpu/ionq/Harmony',
# "operator": "EQUAL",
# "values": ["Status", ...]}], maxResults=25)
while True:
    response = braket.search_quantum_tasks(filters=[{
        'name': 'status',
    "operator": "EQUAL",
    "values": ["CREATED"]}], maxResults=100)

    print(response)
    print(response['quantumTasks'])
    tasks = response['quantumTasks']
    for task in tasks:
        Arn = task['quantumTaskArn']
        print(Arn)
        braket.cancel_quantum_task(quantumTaskArn=Arn)

    if not Arn:
       break



# response = braket.cancel_quantum_task(quantumTaskArn='arn:aws:braket:us-east-1:229406906664:quantum-task/2540729b-4539-4319-8ac1-39f2d4ff0e4e')

# print(f"Quantum task {response['quantumTaskArn']} is {response['cancellationStatus']}")

# [BraketBackend[Aria 1],
# BraketBackend[Aria 2],
# BraketBackend[Aspen-M-3],
# BraketBackend[Forte 1],
# BraketBackend[Harmony],
# BraketBackend[Lucy],
# BraketBackend[SV1],
# BraketBackend[TN1],
# BraketBackend[dm1]]