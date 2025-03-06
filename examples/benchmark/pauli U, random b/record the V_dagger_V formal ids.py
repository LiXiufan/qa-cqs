# fetch the job ids!!!!!!

import boto3

client = boto3.client("braket")
paginator = client.get_paginator('search_quantum_tasks')

file = open("V_dagger_V.txt", "a")

response = paginator.paginate(filters=[
    {
    'name': 'status',
"operator": "EQUAL",
"values": ["QUEUED"]
    }
], PaginationConfig={
        'MaxItems': 100,
        'StartingToken': 'string'
    }
)

print(response)
for k in response.keys():
    print(k)
print(response['quantumTasks'])
tasks = response['quantumTasks']
for i, task in enumerate(tasks):
    if i > 23:
        Arn = task['quantumTaskArn']
        print(Arn)
        # file.writelines(Arn + "\n")
file.close()

