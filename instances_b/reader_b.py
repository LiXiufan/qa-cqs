from pathlib import Path
import pandas as pd

def read_csv_b(n):
    # Get the working directory
    working_dir = Path.cwd()

    # Move up the directory tree until reaching 'CQS_singapore'
    while working_dir.name != "qa-cqs" and working_dir != working_dir.parent:
        working_dir = working_dir.parent

    # Construct the new file path relative to 'CQS_singapore'
    file_path = working_dir / "instances_b" / (str(n)+"_b_random_circuits.csv")

    # Print the resolved path
    print(f"Resolved file path: {file_path}")

    # Check if the file exists
    if file_path.exists():
        print(f"Reading file: {file_path}")
        data_b = pd.read_csv(file_path)
        return data_b
    else:
        raise FileNotFoundError(f"Error: File not found at {file_path}")
