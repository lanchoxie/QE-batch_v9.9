import os
import sys

#script_dir=sys.path[0]
script_dir="QE-batch"

if len(sys.argv)==1:
    continue_cal=0
else:
    continue_cal=sys.argv[1]

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

type_hpc_out=JOB_func(flat_number)

if type_hpc_out=="slurm":
    sub_method="sbatch"
    del_method="scancel"
elif type_hpc_out=="pbs":
    sub_method="qsub"
    del_method="qdel"


depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for dirname in dirnames:
            str_in_dir.append(dirname)
files_id=[]
for i in str_in_dir:
    jj=os.popen(f"python {script_dir}/job_on_hpc_id.py {i}").readlines()[0]
    jj="".join([i for i in jj.strip("\n") if i.isdigit()])
    if len(jj)>0:
        print(i,int(jj))
        files_id.append([i,int(jj)])
print(f"Running on {type_hpc_out} job system, use {del_method}")
if continue_cal=="1" and len(files_id)>0:
    yes_to_all_continue=""
    yes_to_all_error=""
    yes_to_all_error1=""
    for i in range(len(files_id)):
        if (yes_to_all_error==""):
            print("**********************************************************")
            print(f"handling the outlier {files_id[i][0]}:{files_id[i][1]}\n")
            print("Would you want to cancel it?")
            #print("would you like me to continue cal for you?")
            print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
            continue_error=input(">>>>>>")
            if continue_error.lower()=="y":
                os.system(f"{del_method} {files_id[i][1]}") 
                print(f"{del_method} {files_id[i][1]}") 

            elif continue_error.lower()=="q":
                yes_to_all_error="q"
            elif continue_error.lower()=="a":
                yes_to_all_error="a"
                os.system(f"{del_method} {files_id[i][1]}") 
                print(f"{del_method} {files_id[i][1]}") 
                
            elif continue_error.lower()=="n":
                print("skip this file")
            else:
                print("unknow input,skip this loop")
        elif (yes_to_all_error=="a"):
            os.system(f"{del_method} {files_id[i][1]}") 
            print(f"{del_method} {files_id[i][1]}") 
else:
    print("There is no job running in this directory!")
