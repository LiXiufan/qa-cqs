import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run simulation for given L and K.")
parser.add_argument("--L", type=int, required=True, help="Value of L")
parser.add_argument("--K", type=int, required=True, help="Value of K")
args = parser.parse_args()

L = args.L
K = args.K

T = np.array([L, K, L * K])

output_filename = f"output_L{L}_K{K}.npy"
np.save(output_filename, T)

print(f"Save output to {output_filename}")




