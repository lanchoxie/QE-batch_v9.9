import os
import sys

dir_check=sys.argv[1]

root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
code_dir=sys.path[0]
############job_script_settings##########
###You can modify the FLAT_SAVE file in QE-batch/flat.save/ to manually add default settings of your flats
def read_files(infiles):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "#" in lines:
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

flat_save=read_files("%s/flat.save/FLAT_SAVE"%code_dir)
#flat save has files format like below:
#1  zf_normal  pbs  52
#flat_number  queue_name  type_flat  ppn_num_defaut

flat_info=open("%s/flat.save/FLAT_INFO"%code_dir).readlines()
for lines in flat_info:
    if lines.find("flat_number=")!=-1:
        flat_number=int(lines.split("flat_number=")[-1].split("#")[0])
    if lines.find("node_num=")!=-1:
        node_num=int(lines.split("node_num=")[-1].split("#")[0])
    if lines.find("ppn_num")!=-1:
        ppn_num_man=int(lines.split("ppn_num=")[-1].split("#")[0])
        if ppn_num_man==0:
            ppn_set=0
        elif ppn_num_man!=0:
            ppn_set=1
    if lines.find("wall_time=")!=-1:
        wall_time=lines.split("wall_time=")[-1].split("#")[0].strip("\n")


def JOB_func(flat_number):
    flat_ind=[int(i[0]) for i in flat_save].index(flat_number)
    type_flat=flat_save[flat_ind][1]
    type_hpc=flat_save[flat_ind][2]
    ppn_num=int(flat_save[flat_ind][3])
    return type_hpc

dir_file=(os.popen("pwd").read())
dir=max(dir_file.split('\n'))

def split_lines(infiles):
    f1=infiles
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "JOBID" in lines:
            continue
        if "Job ID" in lines:
            continue
        if "----" in lines:
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
    return read_data

if JOB_func(flat_number)=="pbs":
    job_on_hpc=split_lines(os.popen("qstat").readlines())
    wait_node_list=[]
    run_node_list=[]
    for i in job_on_hpc:
        if i[4]=="Q":
            wait_node_list.append(i[0].split(".node1")[0])
        if i[4]=="R":
            run_node_list.append(i[0].split(".node1")[0])
    wait_file_list=[]
    run_file_list=[]
    working_file=[]
    running_file=[]
    for i in wait_node_list:
        all_data=os.popen("qstat -f %s"%i).readlines()
        read_start=0
        working_file_raw=""
        for j in range(len(all_data)):
            if all_data[j].find("PBS_O_WORKDIR")!=-1:
                read_start=1
            if all_data[j].find("PBS_O_HOST")!=-1:
                read_start=2
            if read_start==1:
                #print(all_data[j])
                working_file_raw+=all_data[j].split("\n")[0].strip("\t").strip(",")
            if read_start==2:
                read_start=0
                if all_data[j].split("\n")[0].strip("\t").strip()[:len("PBS_O_HOST")]=="PBS_O_HOST":
                    donothing=1
                else:
                    working_file_raw+=all_data[j].split(",PBS_O_HOST")[0].strip("\t")
        working_file.append(working_file_raw)
    for i in range(len(working_file)):
        working_file[i]=working_file[i].split("PBS_O_WORKDIR=")[-1]

    for i in run_node_list:
        all_data=os.popen("qstat -f %s"%i).readlines()
        read_start=0
        running_file_raw=""
        for j in range(len(all_data)):
            if all_data[j].find("PBS_O_WORKDIR")!=-1:
                read_start=1
            if all_data[j].find("PBS_O_HOST")!=-1:
                read_start=2
            if read_start==1:
                #print(all_data[j])
                running_file_raw+=all_data[j].split("\n")[0].strip("\t").strip(",")
            if read_start==2:
                read_start=0
                if all_data[j].split("\n")[0].strip("\t").strip()[:len("PBS_O_HOST")]=="PBS_O_HOST":
                    donothing=1
                else:
                    running_file_raw+=all_data[j].split(",PBS_O_HOST")[0].strip("\t")
        running_file.append(running_file_raw)
    for i in range(len(running_file)):
        running_file[i]=running_file[i].split("PBS_O_WORKDIR=")[-1]

elif JOB_func(flat_number)=="slurm":
    job_on_hpc=split_lines(os.popen("squeue").readlines())
    wait_node_list=[]
    run_node_list=[]
    for i in job_on_hpc:
        if i[4]=="PD":
            wait_node_list.append(i[0])
        if i[4]=="R":
            run_node_list.append(i[0])
        
    wait_file_list=[]
    working_file=[]
    run_file_list=[]
    running_file=[]

    for i in wait_node_list:
        all_data=os.popen("scontrol show job %s"%i).readlines()
        read_start=0
        working_file_raw=""
        for j in range(len(all_data)):
            if all_data[j].find("WorkDir")!=-1:
                read_start=1
            if all_data[j].find("StdErr")!=-1:
                read_start=0
            if read_start==1:
                #print(all_data[j])
                working_file_raw+=all_data[j].split("\n")[0].strip("\t").strip("./")
        working_file.append(working_file_raw)
    for i in range(len(working_file)):
        working_file[i]=working_file[i].split("WorkDir=")[-1]

    for i in run_node_list:
        all_data=os.popen("scontrol show job %s"%i).readlines()
        read_start=0
        running_file_raw=""
        for j in range(len(all_data)):
            if all_data[j].find("WorkDir")!=-1:
                read_start=1
            if all_data[j].find("StdErr")!=-1:
                read_start=0
            if read_start==1:
                #print(all_data[j])
                running_file_raw+=all_data[j].split("\n")[0].strip("\t").strip("./")
        running_file.append(running_file_raw)
    for i in range(len(running_file)):
        running_file[i]=running_file[i].split("WorkDir=")[-1]
#print(working_file)
os.chdir(dir_check)
read_dir=max(os.popen("pwd").read().split("\n"))
#print(read_dir)
if read_dir in working_file:
    #print("waiting on line")
    print(wait_node_list[working_file.index(read_dir)])
elif read_dir in running_file:
    #print("running")
    print(run_node_list[running_file.index(read_dir)])
else:
    print("Not here")
os.chdir(dir)
