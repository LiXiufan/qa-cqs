import csv
import pandas as pd
from numpy import array, zeros

fine_name = "hardware_result_Q.csv"
Q_noisy = array(pd.read_csv(fine_name).values.tolist())

Q_noiseless = array([[ 3.98710000e+01, -1.49624972e-01, -1.20480736e-14, -1.96917482e-15,
  -2.06790141e-16, -4.67625938e-17,  2.65927680e+01,  2.22044605e-16],
 [-1.49624972e-01,  3.98710000e+01, -3.44724249e-17, -1.03899112e-14,
   4.67625938e-17, -5.81934501e-16, -0.00000000e+00,  2.65927680e+01],
 [-1.20480736e-14, -3.44724249e-17,  3.98710000e+01,  1.49624972e-01,
  -2.65927680e+01,  0.00000000e+00,  2.87677659e-15,  1.66237024e-15],
 [-1.96917482e-15, -1.03899112e-14,  1.49624972e-01,  3.98710000e+01,
  -2.22044605e-16, -2.65927680e+01, -1.66237024e-15,  2.15455986e-15],
 [ 2.06790141e-16,  4.67625938e-17, -2.65927680e+01, -2.22044605e-16,
   3.98710000e+01, -1.49624972e-01, -1.20480736e-14, -1.96917482e-15],
 [-4.67625938e-17,  5.81934501e-16,  0.00000000e+00, -2.65927680e+01,
  -1.49624972e-01,  3.98710000e+01, -3.44724249e-17, -1.03899112e-14],
 [ 2.65927680e+01, -0.00000000e+00, -2.87677659e-15, -1.66237024e-15,
  -1.20480736e-14, -3.44724249e-17,  3.98710000e+01,  1.49624972e-01],
 [ 2.22044605e-16,  2.65927680e+01,  1.66237024e-15, -2.15455986e-15,
  -1.96917482e-15, -1.03899112e-14,  1.49624972e-01,  3.98710000e+01]])

def Q_to_V_dagger_V(Q):
    n = len(Q)
    R = Q[:int(n/2), :int(n/2)]
    I = Q[int(n/2):, :int(n/2)]
    V_dagger_V = R + I * 1j
    return V_dagger_V

V_dagger_V_noiseless = Q_to_V_dagger_V(Q_noiseless)
V_dagger_V_noisy = Q_to_V_dagger_V(Q_noisy)

file_record_name = "heatmap_data.txt"
file_record = open(file_record_name, "a")
for i in range(len(V_dagger_V_noiseless)):
    for j in range(i, len(V_dagger_V_noiseless)):
        error = abs(V_dagger_V_noiseless[i][j] - V_dagger_V_noisy[i][j])
        file_record.writelines([str(j + 1)+" "+str(i + 1)+" "+str(error), '\n'])


# file_record_name = "heatmap_data_Q.txt"
# file_record = open(file_record_name, "a")
# for i in range(len(Q_noiseless)):
#     for j in range(len(Q_noiseless)):
#         error = abs(Q_noiseless[i][j] - Q_noisy[i][j])
#         file_record.writelines([str(j + 1)+" "+str(i + 1)+" "+str(error), '\n'])






