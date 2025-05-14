import csv
import pandas as pd
from numpy import array, zeros

fine_name = "hardware_result_Q.csv"
Q_noisy = array(pd.read_csv(fine_name).values.tolist())

Q_noiseless = array([[ 1.00066000e+01, -4.10838030e-17, -5.57520000e+00,  5.89528426e-18,
   6.90980606e-16,  4.44089210e-16, -1.37486689e-15, -4.44089210e-16],
 [-4.10838030e-17,  1.00066000e+01, -6.68964883e-16, -5.57520000e+00,
  -4.44089210e-16,  8.48487947e-16, -4.44089210e-16, -1.13982712e-15],
 [-5.57520000e+00, -6.68964883e-16,  1.00066000e+01, -1.96792582e-16,
   1.37486689e-15,  4.44089210e-16,  8.76299033e-17,  1.02464703e-15],
 [ 5.89528426e-18, -5.57520000e+00, -1.96792582e-16,  1.00066000e+01,
   4.44089210e-16,  1.13982712e-15, -1.02464703e-15, -6.90980606e-16],
 [-6.90980606e-16, -4.44089210e-16,  1.37486689e-15,  4.44089210e-16,
   1.00066000e+01, -4.10838030e-17, -5.57520000e+00,  5.89528426e-18],
 [ 4.44089210e-16, -8.48487947e-16,  4.44089210e-16,  1.13982712e-15,
  -4.10838030e-17,  1.00066000e+01, -6.68964883e-16, -5.57520000e+00],
 [-1.37486689e-15, -4.44089210e-16, -8.76299033e-17, -1.02464703e-15,
  -5.57520000e+00, -6.68964883e-16,  1.00066000e+01, -1.96792582e-16],
 [-4.44089210e-16, -1.13982712e-15,  1.02464703e-15,  6.90980606e-16,
   5.89528426e-18, -5.57520000e+00, -1.96792582e-16,  1.00066000e+01]])

# def Q_to_V_dagger_V(Q):
#     n = len(Q)
#     R = Q[:int(n/2), :int(n/2)]
#     I = Q[int(n/2):, :int(n/2)]
#     V_dagger_V = R + I * 1j
#     return V_dagger_V
#
# V_dagger_V_noiseless = Q_to_V_dagger_V(Q_noiseless)
# V_dagger_V_noisy = Q_to_V_dagger_V(Q_noisy)
#
# file_record_name = "heatmap_data.txt"
# file_record = open(file_record_name, "a")
# for i in range(len(V_dagger_V_noiseless)):
#     for j in range(i, len(V_dagger_V_noiseless)):
#         error = abs(V_dagger_V_noiseless[i][j] - V_dagger_V_noisy[i][j])
#         file_record.writelines([str(i + 1)+" "+str(j + 1)+" "+str(error), '\n'])


file_record_name = "heatmap_data_Q.txt"
file_record = open(file_record_name, "a")
for i in range(len(Q_noiseless)):
    for j in range(len(Q_noiseless)):
        error = abs(Q_noiseless[i][j] - Q_noisy[i][j])
        file_record.writelines([str(j + 1)+" "+str(i + 1)+" "+str(error), '\n'])






