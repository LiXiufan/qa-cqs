import time
from pprint import pprint
from braket.aws import AwsQuantumJob, AwsSession
from braket.jobs.image_uris import Framework, retrieve_image
import matplotlib.pyplot as plt

job = AwsQuantumJob.create(
    device="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    source_module="depth6.py",
    entry_point="depth6:main",
    wait_until_complete=False,
    job_name="cqs-example3-depth6-" + str(int(time.time())),
    image_uri=retrieve_image(Framework.BASE, AwsSession().region)
)

# arn:aws:braket:us-east-1::device/qpu/ionq/Harmony