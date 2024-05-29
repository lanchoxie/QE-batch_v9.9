import os
import sys
import time
import subprocess as sb

loop_per_dir=20
time_per_loop=3600 #second
debug=0

start_time =time.time()
#make sure the QE-batch software package was named containing "QE-batch" string
#time_schedule=[['LiNiO2-331_NCMT_5', 'relax'], ['LiNiO2-331_NCMT_4', 'relax']]
#time_schedule=[['LiNiO2-331_NCMT_5', 'pdos'], ['LiNiO2-331_NCMT_4', 'pdos'],['LiNiO2-331_NCMT_5', 'BaderCharge'], ['LiNiO2-331_NCMT_4', 'BaderCharge']]
#time_schedule=[['LiNiO2-331_NCMT_5', 'pdos'], ['LiNiO2-331_NCMT_4', 'pdos'],['LiNiO2-331_NCMT_5', 'BaderCharge'], ['LiNiO2-331_NCMT_4', 'BaderCharge']]
#time_schedule=[["LiNiO2_331_NCMT_1-3","scf"],["LiNiO2_331_NCMT_1-3/delete_O","relax"]]
#time_schedule=[["tst1","scf"],["original_cell","relax"],["tst1","scf"],["tst2","scf"],["original_slab","relax"]]        #[[dir1,mode1],[dir2,mode2],...]
#time_schedule=[["LiNiO2_331_NCMT_4131-exhange_file-gamma","relax"],["LiNiO2_331_NCMT_4131-exhange_file-221kp","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.4","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.5","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.6","relax"],["LiNiO2_331_NCMT_4131-exhange_file-local_TF-221kp-beta-0.3","relax"],["LiNiO2_331_NCMT_4131-diff-str-vc-plain-221kp-beta-0.3","relax"],["LiNiO2_331_NCMT_diff-con-vc-plain-221kp-beta-0.3","relax"]]        #[[dir1,mode1],[dir2,mode2],...]
#time_schedule=[["LiNiO2_331_NCMT_4131-exhange_file-gamma","relax"],["LiNiO2_331_NCMT_4131-exhange_file-221kp","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.4","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.5","relax"],["LiNiO2_331_NCMT_4131-exhange_file-beta-0.6","relax"],["LiNiO2_331_NCMT_4131-exhange_file-local_TF-221kp-beta-0.3","relax"],["LiNiO2_331_NCMT_diff-con-vc-plain-221kp-beta-0.3","relax"]]        #[[dir1,mode1],[dir2,mode2],...]
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
def judge_process(code_dir,mode,state,loop,count):
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
        return f"ALL FILES DONE!! DONE: {len(read_datas)}",len(done_file)
    if state==True:
        #print("return")
        return "LOOP DONE!!! DONE: %s Not Done: %s"%(len(done_file),-1*(len(done_file)-len(read_datas))),len(done_file)
    elif state==False and (loop-count*loop_per_dir>=loop_per_dir):
        #print("ALL DONE!!!")
        return "LOOP DONE!!! DONE: %s Not Done: %s"%(len(done_file),-1*(len(done_file)-len(read_datas))),len(done_file)
    elif state==False and (loop-count*loop_per_dir<loop_per_dir) and (loop-count*loop_per_dir>=0):
        return f"rate of process: %s/%s LOOP TIME:{loop-count*loop_per_dir+1}"%(len(done_file),len(read_datas)),len(done_file)
    elif state==False and (loop-count*loop_per_dir<0):
        return f"rate of process: %s/%s Not in LOOP"%(len(done_file),len(read_datas)),len(done_file)
    else:
        return f"rate of process: %s/%s LOOP TIME:{loop-count*loop_per_dir+1}"%(len(done_file),len(read_datas)),len(done_file)

def cal_func(insert=None,file_states=None,QE_dir=None):
    process_li=[]
    err_mess=[]
    if file_states=="None":
        print(f"python {QE_dir}/run-all.py {insert[1]}")
        if debug==0:
            jj=os.popen(f"python {QE_dir}/run-all.py {insert[1]}").readlines()
    elif file_states=="Half":
        jj=os.popen(f"python {QE_dir}/read_relax_E_for_UI.py {insert[1]}").readlines()
        if debug==0:
            jj=os.popen(f"python {QE_dir}/run-all.py {insert[1]} pv_result_out").readlines()
    else:
        #print(f"{insert}||python {QE_dir}/read_relax_E_for_UI.py {insert[1]}")
        jj=os.popen(f"python {QE_dir}/read_relax_E_for_UI.py {insert[1]}").readlines()
        read_datas=read_files("pv_result_out")
        #print(read_datas)
        for i in range(len(read_datas)):
            if read_datas[i][-1].find("running")!=-1:
                cxk=sb.run(["python",f"{QE_dir}/judge_scf_conv.py",read_datas[i][0],"relax"],text=True,stderr=sb.PIPE,stdout=sb.PIPE)
                if cxk.stderr:
                    err_mess.append(f'Error in {read_datas[i][0]}: {cxk.stderr}')
                for j in cxk.stdout.splitlines():
                    if "Modify" in j:
                        process_li.append(f"Modify scf in {read_datas[i][0]}")

                cxk2=sb.run(["python",f"{QE_dir}/judge_relax_conv.py",read_datas[i][0]],text=True,stderr=sb.PIPE,stdout=sb.PIPE)
                if cxk2.stderr:
                    err_mess.append(f'Error in {read_datas[i][0]}: {cxk2.stderr}')
                for j in cxk2.stdout.splitlines():
                    if "Modify" in j:
                        process_li.append(f"Modify relax in {read_datas[i][0]}")

            if (read_datas[i][-1].find("not converged")!=-1)|(read_datas[i][-1].find("out of timing")!=-1)|(read_datas[i][-1].find("ERROR")!=-1):
                #print("python %s/continue_cal.py %s relax"%(code_dir,read_datas[i][0]))
                cxk=sb.run(["python",f"{QE_dir}/judge_scf_conv.py",read_datas[i][0],"relax"],text=True,stderr=sb.PIPE,stdout=sb.PIPE)
                if cxk.stderr:
                    err_mess.append(f'Error in {read_datas[i][0]}: {cxk.stderr}')
                for j in cxk.stdout.splitlines():
                    if "Modify" in j:
                        process_li.append(f"Modify scf in {read_datas[i][0]}")

                cxk2=sb.run(["python",f"{QE_dir}/judge_relax_conv.py",read_datas[i][0]],text=True,stderr=sb.PIPE,stdout=sb.PIPE)
                if cxk2.stderr:
                    err_mess.append(f'Error in {read_datas[i][0]}: {cxk2.stderr}')
                for j in cxk2.stdout.splitlines():
                    if "Modify" in j:
                        process_li.append(f"Modify relax in {read_datas[i][0]}")

                cxk=os.popen("python %s/continue_cal.py %s relax"%(QE_dir,read_datas[i][0])).readlines()
                process_li.append("python %s/continue_cal.py %s relax"%(QE_dir,read_datas[i][0]))

            elif (read_datas[i][-1].find("No Calculation")!=-1):
                cxk=os.popen(f"python %s/replace.py %s.vasp {insert[1]} 0 0"%(QE_dir,read_datas[i][0])).readlines()
                process_li.append(f"python %s/replace.py %s.vasp {insert[1]} 0 0"%(QE_dir,read_datas[i][0]))

    return f"calculated in directory :{insert[0]} \n",process_li

all_dir_done=False
loop_in=0
while all_dir_done==False:
    tot_done_file=0
    buffer_loop=0
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
            #print(QE_script_dir[0],i[1],i[2],loop_in,j)
            state_of_file,dir_done_i=judge_process(QE_script_dir[0],i[1],i[2],loop_in,j)
            tot_done_file+=dir_done_i
            print("\t"+state_of_file)
            if state_of_file.find("DONE")!=-1:
                i[2]=True
            if state_of_file.find("ALL")!=-1:
                loop_in+=loop_per_dir
                buffer_loop+=loop_per_dir
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
    loop_in+=1
    loop_in-=buffer_loop
    #print(time_schedule)
    print("Total loops:",loop_in)
    print(f"Total done files count is {tot_done_file} now!")
    time.sleep(time_per_loop)
    print("\n********************************\n************************************\n")

print("ALL DIR LOOP DONE!!!")
print(f"Total Time Cost: {hour}小时{minute}分钟{second}秒")
