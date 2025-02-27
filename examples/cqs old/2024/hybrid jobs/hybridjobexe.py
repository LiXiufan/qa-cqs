import time
from pprint import pprint
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.image_uris import Framework, retrieve_image
import matplotlib.pyplot as plt

job = AwsQuantumJob.create(
    device="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    source_module="example4.py",
    entry_point="example4:main",
    wait_until_complete=False,
    job_name="cqs-example4-depth7-" + str(int(time.time())),
    image_uri=retrieve_image(Framework.BASE, AwsSession().region),
)

# arn:aws:braket:us-east-1::device/qpu/ionq/Harmony
