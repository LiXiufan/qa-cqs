text_file = open('shotfrugalData.txt','r')
text_file2 = open('shotfrugalDataPlot.txt','a')
LINES = text_file.readlines()

for l in range(5, len(LINES), 8):
    a = LINES[l]
    b = a.split(':')[1].strip()
    c = LINES[l+1]
    d = c.split(':')[1].strip()
    text_file2.writelines([b+" "+d, '\n'])