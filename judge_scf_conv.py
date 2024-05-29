import os
import sys
import re

sys.path.append(sys.path[0])
from libs.modify_file import modify_parameter

tot_dir=sys.argv[1].strip("/")
mode=sys.argv[2]
str_name=tot_dir.split("/")[-1]
scf_not_conv_time=2 # scf not convergence time 

code_path=sys.path[0]

process=1

input_name=f"{tot_dir}/in_{mode}_{str_name}"
output_name=f"{tot_dir}/out_{mode}_{str_name}"
print(output_name)

if os.path.isfile(f"{code_path}/bcon.py")==1:
    print("bcon.py found")
    find_bcon=1
if find_bcon==0:
    print("bcon.py not found,plz download the full package of QE-batch!")

dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))
type_hpc_out=os.popen(f"python {code_path}/create_job_script.py {str_name} {mode}").readline().strip("\n").strip().split("###")[-1]
if type_hpc_out=="slurm":
    sub_method="sbatch"
    del_method="scancel"
elif type_hpc_out=="pbs":
    sub_method="qsub"
    del_method="qdel"

replace_line=open("%s/replace.py"%code_path).readlines()
for i in replace_line:
    if i.find("JOB=")!=-1:
        JOB_name=i.split("JOB=")[-1].split("#")[0].split("\"")[1]

if os.path.isfile(output_name)==0:
    raise ValueError("Not Running!")

input_cont=open(input_name).readlines()
E_must_conv=True
E_max_step=100
for line in input_cont:
    if "scf_must_converge" in line and "!" not in line and "false" in line:
        E_must_conv=False
    if "electron_maxstep" in line and "!" not in line:
        E_max_step=int(line.split("=")[-1].strip("\n").strip(",").strip())

def count_consecutive(arr,E_max_step):
    count = 0
    if arr[-1] != E_max_step and arr[-2] != E_max_step:
        return 0
    for i in range(len(arr) - 1, -1, -1):  # 从数组的末尾开始遍历
        if arr[i] == E_max_step:
            count += 1  # 如果元素是max_step，增加计数器
        else:
            break  # 如果元素不是max_step，结束循环
    return count


def scf_conv_judge():
    out_scf_conv=os.popen(f"grep \"convergence has\" {output_name}").readlines()
    if len(out_scf_conv)<scf_not_conv_time:
        raise ValueError("Not enough data!")
    conv_steps=[int(i.split("in")[-1].split("itera")[0].strip()) for i in out_scf_conv]      
    conv_not_achiev=count_consecutive(conv_steps,E_max_step)
    return conv_not_achiev

mix_beta=os.popen(f"grep mixing_beta {input_name}").readlines()
mixing_beta=0.7
for i in mix_beta:
    if "!" not in i:
        mixing_beta=float(i.strip("\n").strip(",").split("=")[-1].strip())
conv_not_achiev=(scf_conv_judge())    
print(conv_not_achiev)
job_id=os.popen(f"python {code_path}/job_on_hpc_id.py {tot_dir}").readline().strip("\n").strip()
job_state=os.popen(f"python {code_path}/job_on_hpc.py {tot_dir}").readline().strip("\n").strip()
if E_must_conv==False:
    if conv_not_achiev>scf_not_conv_time:
        print(f"!!!Modify\n***********************{job_state}")
        if job_id!="Not here" and job_state!="waiting on line":
            print(f"{del_method} {job_id}")
            if process==1:
                os.system(f"{del_method} {job_id}")
            print(f"change into {tot_dir}")
            if process==1:
                os.chdir(tot_dir)
            print(f"modify in_{mode}_{str_name}, beta={mixing_beta*0.8:.3}")
            if process==1:
                modify_parameter("./",f"{str_name}","mixing_beta",f"{mixing_beta*0.8:.3}","&electrons","    ")
            print(f"mv out_{mode}_{str_name} out_buffer")
            if process==1:
                os.system(f"python {code_path}/bcon.py out_{mode}_{str_name} in_{mode}_{str_name}")
                os.system(f"mv out_{mode}_{str_name} out_buffer")
            print(f"{sub_method} {JOB_name}")
            if process==1:
                os.system(f"{sub_method} {JOB_name}")
            print(f"change into {current_dir}")
            if process==1:
                os.chdir(current_dir)
            #modify_parameter(tot_dir,f"{str_name}","mixing_beta",0.2,"&electrons","    ")
        elif job_id=="Not here":
            print(f"change into {tot_dir}")
            if process==1:
                os.chdir(tot_dir)
            print(f"modify in_{mode}_{str_name}, beta={mixing_beta*0.8:.3}")
            if process==1:
                modify_parameter("./",f"{str_name}","mixing_beta",f"{mixing_beta*0.8:.3}","&electrons","    ")
            print(f"mv out_{mode}_{str_name} out_buffer")
            if process==1:
                os.system(f"python {code_path}/bcon.py out_{mode}_{str_name} in_{mode}_{str_name}")
                os.system(f"mv out_{mode}_{str_name} out_buffer")
            #print(f"{sub_method} {JOB_name}")
            print(f"change into {current_dir}")
            if process==1:
                os.chdir(current_dir)
    else:
        print(f"!!!Nothing to be done\n***********************{job_state}")
