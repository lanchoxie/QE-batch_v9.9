import os
import sys
import time



stress_shr=0.5
loop_time=600
sub_lim=20 #submission limit
max_lim=15 #submit time limit for single job
script_dir=sys.path[0]

start_time =time.time()

depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if ".vasp" in filename:
                str_in_dir.append(filename.split(".vasp")[0])

loop_count=0
all_done=0
while all_done==0:

    vasp_list=[f.split(".vasp")[0] for f in os.listdir(".") if ".vasp" in f]
    count=[]
    for i in vasp_list:
        print(f"{vasp_list.index(i)}","\r",end="")
        f_list=[f for f in os.listdir(i) if "out_relax" in f]
        count.append([i,len(f_list)])

    file_infos=[]
    for i in str_in_dir:
        if not os.path.exists(i):
            continue
        if not os.path.isfile(f"{i}/out_relax_{i}"):
            stress_info=[i,999]
            file_infos.append(stress_info)
            continue
        jj1=os.popen(f"grep \"total   stress\" {i}/out_relax_{i}").readlines()
        if len(jj1)>0:
            jj=jj1[-1]
            stress_info=[i,float(jj.split()[-1])]
        else:
            stress_info=[i,999]
        #print(stress_info)
        file_infos.append(stress_info)
    
    print("reading relax results...")
    ak=os.popen(f"python {script_dir}/read_relax_E_for_UI.py relax").readlines()
    results_pv=open("pv_result_out").readlines()
    for v,i in enumerate(results_pv):
        print(f"reading stress {v}/{len(results_pv)}","\r",end="")
        state_info=[i.split()[0],i.split()[-1]]
        if state_info[0] in [x[0] for x in file_infos]:
            file_infos[[x[0] for x in file_infos].index(state_info[0])].append(state_info[1])
    for i in file_infos:
        #print(i)
        if abs(i[1])>stress_shr and i[-1].find("DONE")!=-1:
            i.append("stress not converge")
        elif i[-1].find("timing")!=-1:
            i.append("stress not converge")
        elif i[-1].find("ERROR")!=-1:
            i.append("stress not converge")
        elif i[-1].find("Calculation")!=-1:
            i.append("stress not converge")
        elif i[-1].find("converged")!=-1:
            i.append("stress not converge")
        else:
            i.append(i[-1])
        if i[0] in [x[0] for x in count]:
            if count[[x[0] for x in count].index(i[0])][1]>max_lim and i[-1].find("DONE")==-1:
                i[3]=f"out of limit {max_lim}:Pause"
                i[1]=1000
        i.insert(-1,f"output file number: {count[[x[0] for x in count].index(i[0])][1]}")
        print(i)
    done_file=[]
    run_file=[]
    wait_file=[]
    not_con_file=[]
    pause_file=[]
    for i in file_infos:
        if i[-1]=="stress not converge":
            not_con_file.append(i)
        elif i[-1].find(f"out of limit {max_lim}:Pause")!=-1:
            pause_file.append(i)
        elif i[-1].find("wait")!=-1:
            wait_file.append(i)
        elif i[-1].find("run")!=-1:
            run_file.append(i)
        elif i[-1].find("DONE")!=-1:
            done_file.append(i)
    current_run_job=len(run_file)+len(wait_file)
    if current_run_job<sub_lim:
        sub_job_num=sub_lim-current_run_job
        
        for i in range(min(sub_job_num,len(not_con_file))):
            os.system(f"python {script_dir}/continue_cal.py {not_con_file[i][0]} relax")
            print(f"python {script_dir}/continue_cal.py {not_con_file[i][0]} relax")

    print("*"*20+"\nThe files below has pause calculation:")
    for i in pause_file:
        print(i[0])
    
    loop_count+=1
    print("*"*20,f"LOOP TIME: {loop_count}","*"*20)
    if current_run_job<sub_lim:
        print(f"Sub {min(sub_job_num,len(not_con_file))} at this LOOP")
    else:
        print(f"Running job number at limit of {sub_lim}")
    print(f"DONE:{len(done_file)} RUN:{len(run_file)} WAIT:{len(wait_file)} NOT_CONV:{len(not_con_file)} PAUSE:{len(pause_file)}")
    print("*"*60)
    end_time=time.time()
    run_time = round(end_time-start_time)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    if len(done_file)==len(file_infos):
        all_done=1
        break
    if all_done==False:
        print(all_done,len(done_file)+len(pause_file),len(file_infos))
        if (len(done_file)+len(pause_file))==len(file_infos):
            print(f"Running: {hour}小时{minute}分钟{second}秒")
            yes_to_all_error=""
            print("Those files are paused calculation:")
            for i in pause_file:
                print(i)
            for i in range(len(pause_file)):
                exe_line=f"python {script_dir}/check_vc_continue_cal.py {pause_file[i][0]}"
                if (yes_to_all_error==""):
                    print("**********************************************************")
                    print(f"handling the pause_file {pause_file[i][0]}\n")
                    print("Wanna restart the pause calculation? plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                    continue_error=input(">>>>>>")
                    if continue_error.lower()=="y":
                        os.system(exe_line)
                        print(exe_line)
                    elif continue_error.lower()=="q":
                        yes_to_all_error="q"
                    elif continue_error.lower()=="a":
                        yes_to_all_error="a"
                        os.system(exe_line)
                        print(exe_line)
                 
                    elif continue_error.lower()=="n":
                        print("skip this file")
                    else:
                        print("unknow input,skip this loop")
                elif (yes_to_all_error=="a"):
                    os.system(exe_line)
                    print(exe_line)
        else:
            print(f"Running: {hour}小时{minute}分钟{second}秒")
            time.sleep(loop_time)
print(f"ALL {len(done_file)} FILES DONE!!")
print(f"Total Time Cost: {hour}小时{minute}分钟{second}秒")
