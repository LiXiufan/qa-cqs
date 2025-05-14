from numpy import real
text_file = open('shotfrugalData.txt','r')
text_file2 = open('shotfrugalErrorPlot.txt','a')
LINES = text_file.readlines()

# for l in range(5, len(LINES), 11):
#     a = LINES[l]
#     exact_result = real(complex(a.split(':')[1].strip()))
#     b = LINES[l+1]
#     ua_result = real(complex(b.split(':')[1].strip()))
#     c = LINES[l+2]
#     wa_result = real(complex(c.split(':')[1].strip()))
#     text_file2.writelines([str(exact_result)+" "+str(ua_result)+" "+str(wa_result), '\n'])

UA = []
WA = []
for l in range(8, len(LINES), 11):
    a = LINES[l]
    UA.append(float(a.split(':')[1].strip()))
    b = LINES[l+1]
    WA.append(float(b.split(':')[1].strip()))
ua_ave = sum(UA)/len(UA)
wa_ave = sum(WA)/len(WA)
text_file2.writelines([str(ua_ave)+" "+str(wa_ave), '\n'])