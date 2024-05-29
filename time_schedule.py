import os
import sys
import time


start_time =time.time()
#make sure the QE-batch software package was named containing "QE-batch" string

#time_schedule=[["tst1","scf"],["original_cell","relax"],["tst1","scf"],["tst2","scf"],["original_slab","relax"]]        #[[dir1,mode1],[dir2,mode2],...]
#time_schedule=[["LiNiO2_331_NCM_423-exhange_file","relax"],["LiNiO2_331_NCMT_3132-exhange_file","relax"],["LiNiO2_331_NCM_111-exhange_file","relax"],["LiNiO2_331_NCMT_4131-exhange_file","relax"],["LiNiO2_331_NCM_513-exhange_file","relax"],["LiNiO2_331_NC_21-exhange_file","relax"]]        #[[dir1,mode1],[dir2,mode2],...]
############ Above is what you need to change #################
current_path=sys.path[0]
work_path=current_path+"/"
#print(work_path)

for tasks in time_schedule:
    tasks.append(False)


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
def judge_process(code_dir,mode):
    #error_file=[]
    done_file=[]
    cxk=os.popen("python %s/read_relax_E_for_UI.py %s"%(code_dir,mode)).readlines()
    read_datas=read_files("pv_result_out")
    for i in range(len(read_datas)):
        if (read_datas[i][-1].find("!DONE!")!=-1):
            done_file.append(i)
        elif (read_datas[i][-1].find("out of timing")!=-1)|(read_datas[i][-1].find("ERROR")!=-1):
            pass
            #print(read_datas[i][0])
            #error_file.append(read_datas[i])
            #print("python %s/continue_cal.py %s relax"%(code_dir,read_datas[i][0]))
            #cxk=os.popen("python %s/continue_cal.py %s relax"%(code_dir,read_datas[i][0])).readlines()
    #print("Round:",rounds)
    #print("rate of process: %s/%s"%(len(done_file),len(read_datas)))
    if len(done_file)==len(read_datas):
        #print("ALL DONE!!!")
        return "ALL DONE!!!"
    else:
        return "rate of process: %s/%s"%(len(done_file),len(read_datas))

def cal_func(insert=None,file_states=None,QE_dir=None):
    process_li=[]
    if file_states=="None":
        jj=os.popen(f"python {QE_dir}/run-all.py {insert[1]}").readlines()
    elif file_states=="Half":
        jj=os.popen(f"python {QE_dir}/read_relax_E_for_UI.py {insert[1]}").readlines()
        jj=os.popen(f"python {QE_dir}/run-all.py {insert[1]} pv_result_out").readlines()
    else:
        jj=os.popen(f"python {QE_dir}/read_relax_E_for_UI.py {insert[1]}").readlines()
        read_datas=read_files("pv_result_out")
        for i in range(len(read_datas)):
            if (read_datas[i][-1].find("No Calculation")!=-1)|(read_datas[i][-1].find("not converged")!=-1)|(read_datas[i][-1].find("out of timing")!=-1)|(read_datas[i][-1].find("ERROR")!=-1):
                #print("python %s/continue_cal.py %s relax"%(code_dir,read_datas[i][0]))
                cxk=os.popen("python %s/continue_cal.py %s relax"%(QE_dir,read_datas[i][0])).readlines()
                process_li.append("python %s/continue_cal.py %s relax"%(QE_dir,read_datas[i][0]))
    return f"calculated in directory :{insert[0]} \n",process_li

all_dir_done=False
while all_dir_done==False:
    for j,i in enumerate(time_schedule):
        os.chdir(work_path+i[0])
        print("%d.  Dir:"%(j+1),i[0],"  Mode:",i[1],end='\n')
        QE_script_dir=[]
        file_in_dir=[]
        depth=1
        stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
        for parent,dirnames,filenames in os.walk(stuff):
            if parent[len(stuff):].count(os.sep) < depth:
                for dirname in dirnames:
                    if "QE-batch" in dirname:
                        QE_script_dir.append(dirname)
                for filename in filenames:
                    if ".vasp" in filename:
                        file_in_dir.append(filename.split(".vasp")[0])
        none_exist_file=[]
        for dir_i in file_in_dir:
            if os.path.exists(dir_i)==0:
                none_exist_file.append(dir_i)
        if len(none_exist_file)==len(file_in_dir):
            #print("\tno calculation done before, using run_all.py mode")
            state_of_file="None"
            print("\t"+state_of_file)
        elif (len(none_exist_file)<len(file_in_dir))&(len(none_exist_file)>0):
            #print("\tsome calculation done before, using run_all.py mode pv_result_out")
            state_of_file="Half"
            print("\t"+state_of_file)
        elif len(none_exist_file)==0:
            #print("script dir ",QE_script_dir[0],os.getcwd())
            state_of_file=judge_process(QE_script_dir[0],i[1])
            print("\t"+state_of_file)
            if state_of_file.find("ALL DONE")!=-1:
                i[2]=True
            
        task_done_before = [time_schedule[x][2] for x in range(len(time_schedule)) if x<j]
        task_done_judge = all(task_done_before)
        #print(len(task_done_before),task_done_judge,i[2],i[0])
        if len(task_done_before)!=0:
            if (task_done_judge==True)&(time_schedule[j][2]==False):
                print_out,process_info=cal_func(insert=i,file_states=state_of_file,QE_dir=QE_script_dir[0])
            else:
                pass
        elif len(task_done_before)==0: 
            if time_schedule[j][2]==False:
                print_out,process_info=cal_func(insert=i,file_states=state_of_file,QE_dir=QE_script_dir[0])
            else:
                pass
                
        os.chdir(work_path)

    if print_out==None:
       print_out="No Job Running"
    if len(process_info)==0:
       process_info=["No operation in this turn"]
    print(f"\n************************************\n{print_out}")
    for infos in process_info:
        print(f"\n{infos}")
    all_dir_done=all([j[2] for j in time_schedule])
    end_time=time.time()
    run_time = round(end_time-start_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    if all_dir_done==False:
        print_stuff=f"Running: {hour}小时{minute}分钟{second}秒"
    elif all_dir_done==True:
        break
    print(f"\n*********{print_stuff}*********\n",'\r',end='')
    time.sleep(3600)
print("ALL DIR JOB DONE!!!")
print(f"Total Time Cost: {hour}小时{minute}分钟{second}秒")
