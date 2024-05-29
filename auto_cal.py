import os
import sys
import time

root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
mode=sys.argv[1]
code_dir=sys.path[0]
loop_time=100

def read_files(infiles):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "Direct" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(" ")
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(i)
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data


for rounds in range(1000):
    error_file=[]
    done_file=[]
    cxk=os.popen(f"python %s/read_relax_E_for_UI.py {mode}"%code_dir).readlines()
    read_datas=read_files("pv_result_out")
    for i in range(len(read_datas)):
        if (read_datas[i][-1].find("!DONE!")!=-1):
            done_file.append(i)
        elif (read_datas[i][-1].find("No Calculation")!=-1)|(read_datas[i][-1].find("not converged")!=-1)|(read_datas[i][-1].find("out of timing")!=-1)|(read_datas[i][-1].find("ERROR")!=-1):
            #print(read_datas[i][0])
            error_file.append(read_datas[i])
            print(f"python %s/continue_cal.py %s {mode}"%(code_dir,read_datas[i][0]))
            cxk=os.popen(f"python %s/continue_cal.py %s {mode}"%(code_dir,read_datas[i][0])).readlines()
    print("Round:",rounds)
    print("rate of process: %s/%s"%(len(done_file),len(read_datas)))
    if len(done_file)==len(read_datas):
        print("ALL DONE!!!")
        break
    time.sleep(600)
    
