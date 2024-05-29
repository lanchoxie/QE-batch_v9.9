import numpy
import os
import time
import sys


prefix="BTO"  #the prefix same in your input file

root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
code_dir=sys.path[0]

##default value of mode is relax
str_in_dir_i=sys.argv[1]
mode=sys.argv[2]
print(mode)

dir_file=(os.popen("pwd").read())
dir=max(dir_file.split('\n'))
JOB_name=""
error=0
if os.path.isfile("%s/replace.py"%code_dir)==1:
    replace_line=open("%s/replace.py"%code_dir).readlines()
#get the job_script name:
    for i in replace_line:
        if i.find("JOB=")!=-1:
            JOB_name=i.split("JOB=")[-1].split("#")[0].split("\"")[1]
            #print("JOB_name**************",JOB_name)
        if i.find("#solvation")!=-1:
            solvation_model=int(i.split("#")[0].split("solvation_model=")[-1].strip())
    if (JOB_name==""):
        print("ERROR: replace.py not set correctly!")
        error=1
elif os.path.isfile("%s/replace.py"%code_dir)==0:
    print("plz copy /hpc/data/home/spst/zhengfan/open/replace/replace.py here, otherwise some func does not work! ")
    error=1

file_name=str_in_dir_i
os.chdir("%s/%s/"%(dir,(str_in_dir_i)))
type_hpc_out=os.popen(f"python {code_dir}/create_job_script.py {file_name} {mode}").readline().strip("\n").strip().split("###")[-1]
os.chdir(dir)
sub_method=""
if type_hpc_out=='pbs':
    sub_method="qsub"
elif type_hpc_out=="slurm":
    sub_method="sbatch"

#print("**************************%s %s"%(sub_method,JOB_name))

find_bcon=0
root_dir = os.path.expandvars('$HOME')
#print("%s/bin/bcon.sh"%root_dir)
if os.path.isfile(f"{code_dir}/bcon.py")==1:
    print("bcon.py found")
    find_bcon=1
if find_bcon==0:
    print("bcon.py not found,plz download the full package of QE-batch!")
 #recalculate if out of timing

os.chdir("%s/%s/"%(dir,(str_in_dir_i)))
if mode=="relax":
    os.system(f"python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir_i),(str_in_dir_i)))
    print(f"python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir_i),(str_in_dir_i)))

os.system("%s %s"%(sub_method,JOB_name))
if os.path.isfile("out_%s_%s"%(mode,str_in_dir_i)):
    os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir_i,mode))
os.chdir(dir)
print("%s subed!"%str_in_dir_i)
