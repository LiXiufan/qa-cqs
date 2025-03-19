# fetch the job ids!!!!!!

import boto3

braket_client = boto3.client("braket")
file = open("V_dagger_V.txt", "a")
nextToken = ''
while True:
    response = braket_client.search_quantum_tasks(
        filters=[
            {
                'name': 'status',
                "operator": "EQUAL",
                "values": ["COMPLETED"]
            }
        ],
        maxResults=100,
        nextToken=nextToken
    )

    print(response['quantumTasks'])
    tasks = response['quantumTasks']
    for i, task in enumerate(tasks):
        Arn = task['quantumTaskArn']
        # print(Arn)
        file.writelines(Arn + "\n")
    nextToken = response['nextToken']
file.close()

