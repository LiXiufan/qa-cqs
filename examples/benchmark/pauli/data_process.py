text_file = open('BenchmarkPauliData.txt','r')
text_file2 = open('BenchmarkPauliPlotSource.txt','a')
text_file3 = open('BenchmarkPauliPlotAverage.txt','a')
LINES = text_file.readlines()

D_N = {str(n):[] for n in range(2, 15)}
D_K = {str(K):[] for K in range(2, 15)}
for l in range(0, len(LINES), 8):
    n = LINES[l].split(':')[1].strip()
    K = LINES[l+1].split(':')[1].strip()
    itr = LINES[l+5].split(':')[1].strip().split(', ')[-1][:-1]
    D_N[n].append(int(itr))
    D_K[K].append(int(itr))
    text_file2.writelines([n+" "+K+" "+itr, '\n'])
D_N_ave = {i:str(int(sum(D_N[i])/len(D_N[i]))) for i in D_N.keys() if D_N[i]}
D_K_ave = {i:str(int(sum(D_K[i])/len(D_K[i]))) for i in D_K.keys() if D_K[i]}
for n in D_N_ave.keys():
    text_file3.writelines([n + " " + D_N_ave[n], '\n'])
text_file3.writelines(['\n'])
for K in D_K_ave.keys():
    text_file3.writelines([K + " " + D_K_ave[K], '\n'])
