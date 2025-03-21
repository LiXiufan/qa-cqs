########################################################################################################################
# Copyright (c) Xiufan Li. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Xiufan Li
# Supervisor: Patrick Rebentrost
# Institution: Centre for Quantum Technologies, National University of Singapore
# For feedback, please contact Xiufan at: shenlongtianwu8@gmail.com.
########################################################################################################################

# !/usr/bin/env python3

r"""
   Each time we want to access to AWS, we have to reset our credentials by changing the access key, secret key,
   and tokens. The way to change it is through terminal.
   1. Type cd ~/.aws and Enter
   2. Type ./credentials and Enter. Use the text editor to open the file.
   3. Delete the access key, secret key, and tokens. Log in to the AWS access portal and seek for your group's
   access key, secret key, and tokens. Copy everything and paste them to the `credentials` file.
"""

import boto3

from braket.aws import AwsDevice
from braket.circuits import Circuit
import logging
from botocore.exceptions import ClientError

# Let's use Amazon S3
s3 = boto3.resource('s3')
# Print out bucket names
for bucket in s3.buckets.all():
    print(bucket.name)

# Retrieve the list of existing buckets
s3 = boto3.client('s3')
response = s3.list_buckets()

# Output the bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        if region is None:
            s3_client = boto3.client('s3')
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True
# create_bucket('amazon-braket-prgroup-xiufan')

# arn:aws:iam::225989338317:role/aws-service-role/braket.amazonaws.com/AWSServiceRoleForAmazonBraket
# device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1")
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# Choose S3 bucket to store results
bucket = 'amazon-braket-prgroup-xiufan'# <<Update with your actual bucket name>> #eg: "amazon-braket-unique-aabbcdd"
prefix = "results"
s3_folder = (bucket, prefix)

bell = Circuit().h(0).cnot(0, 1)
print(bell)

task = device.run(bell, s3_folder, shots=2)
print("Measurement Results")
print(task.result().measurement_counts)













