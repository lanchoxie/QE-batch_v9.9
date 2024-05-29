import os
import sys
import time
sys.path.append(sys.path[0])
from libs.modify_file import modify_flat
import time

## Using the following scripts:
#read_relax_E_for_UI.py
#replace.py
#read_bader.py
#qe_out_reader_gnn.py
#get_surrounding_by_elements_multi.py
#separate_sur_atom_info_lst.py
#separate_sur_atom_info.py
#atomic_info_extractor.py
#main.py
#create_loop_schedule.py
#loop_schedule.py

## Modifying the following files:
#replace.py
#flat.save/FLAT_INFO


stress_shr=0.5
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
#print(i)

print("*"*20+"Done Files"+'*'*20)
done_file_list=[]
undone_file_list=[]
for i in file_infos:
    if i[-1].find("DONE")!=-1:
        done_file_list.append(i)
        print(i)
    else:
        undone_file_list.append(i)

print("*"*20+"Not Done Files"+'*'*20)
for i in undone_file_list:
    print(i)

def split_lines(infiles,split_syb,split_mode=None):
    f1=infiles
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "Direct" in lines:
            continue
        if "***" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(split_syb)
        for i in a_bf:
            if len(i)>0:
                if split_mode=="str" or not split_mode:
                    read_data_row.append(str(i.strip()))
                elif split_mode=="float":
                    read_data_row.append(float(i.strip()))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    return read_data

def check_cal_result_done(str_list,mode,sleep_time):
    jj=os.popen(f"python {script_dir}/read_relax_E_for_UI.py {mode}").readlines()
    result=split_lines(open("pv_result_out").readlines(),"\t",split_mode="str")
    #print(result)
    all_done=False
    while not all_done:
        done_f=[]
        run_f=[]
        wait_f=[]
        notcal_f=[]
        jj=os.popen(f"python {script_dir}/read_relax_E_for_UI.py {mode}").readlines()
        result=split_lines(open("pv_result_out").readlines(),"\t",split_mode="str")
        for str_ in result:
            if str_[0] in str_list:
                if str_[-1].find("run")!=-1:
                    run_f.append(str_[0])
                elif str_[-1].find("wait")!=-1:
                    wait_f.append(str_[0])
                elif str_[-1].find("DONE")!=-1:
                    done_f.append(str_[0])
                else:
                    notcal_f.append(str_[0])
                    print(f"python {script_dir}/replace.py {str_[0]}.vasp {mode} 0 0")
                    os.system(f"python {script_dir}/replace.py {str_[0]}.vasp {mode} 0 0")
            else:
                pass
        print(f"{mode}:{len(done_f)}/{len(str_list)}")
        if len(done_f)==len(str_list):
            all_done=True
        else:
            time.sleep(sleep_time)
    print(f"All calculation of {mode} done!")

def batch_command_old(str_list,command,target_file,extra_tail=None):
    
    for i,v in enumerate(str_list):
        judge_file_i=target_file.replace("*",v)
        if not os.path.isfile(judge_file_i):
            if extra_tail==None:
                print(f"python {command} {v}")
                jj=os.popen(f"python {command} {v}").readlines()
            elif extra_tail!=None:
                print(f"python {command} {v} {extra_tail}")
                jj=os.popen(f"python {command} {v} {extra_tail}").readlines()
        else:
            a=judge_file_i.replace("*",v)
            print(f"{a} FOUND!!")

from concurrent.futures import ProcessPoolExecutor, as_completed
# 定义在模块级别的process_subgraph函数
def process_subgraph(v, command, target_file, extra_tail=None):
    judge_file_i = target_file.replace("*", v)
    if not os.path.isfile(judge_file_i):
        cmd = f"python {command} {v}"
        if extra_tail is not None:
            cmd += f" {extra_tail}"
        print(f"Executing command: {cmd}")  # 模拟执行命令
        # 实际环境中应使用subprocess.run([cmd], check=True) 替代 os.popen
        jj = os.popen(cmd).readlines()
    else:
        print(f"File {judge_file_i} FOUND!!")
    return 1

def print_progress(done, total):
    percent_done = done / total * 100
    bar_length = int(percent_done / 100 * 60)
    bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
    print(bar, end='\r')

def batch_command(graphs, command, target_file, extra_tail=None):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_subgraph, subgraph, command, target_file, extra_tail): i for i, subgraph in enumerate(graphs)}
        converted_graphs = [None] * len(graphs)  # 预先分配结果列表
        total_done = 0  # 已完成的任务数量

        for future in as_completed(futures):
            index = futures[future]  # 获取原始图的索引
            result = future.result()
            converted_graphs[index] = result if result is not None else None
            total_done += 1
            print_progress(total_done, len(graphs))  # 打印进度
        print()  # 打印换行以确保进度条之后的输出不会被覆盖
    time.sleep(0.1)


process_str_list=[x[0] for x in done_file_list]
print("*"*40)
print("#"*20+"Data Sampling Process On..."+"#"*20+"\n")

print("*"*20+"1.Starting Calculating pdos..."+"*"*20)
#os.system(f"cp {script_dir}/flat.save/FLAT_INFO {script_dir}/flat.save/FLAT_INFO_orig")
print(f"modify: {script_dir}/","replace.py","sub_script","1","","")
print(f"modify: {script_dir}/flat.save/","FLAT_INFO","ppn_num","4","node_num","")
print(f"modify: {script_dir}/flat.save/","FLAT_INFO","wall_time","\"00:05:00\"","ppn_num","")
modify_flat(f"{script_dir}","replace.py","sub_script","1","","")
modify_flat(f"{script_dir}/flat.save","FLAT_INFO","ppn_num","4","node_num","")
modify_flat(f"{script_dir}/flat.save","FLAT_INFO","wall_time","\"00:05:00\"","ppn_num","")
check_cal_result_done(process_str_list,"pdos",10)

print("*"*20+"2.Starting Calculating BaderCharge..."+"*"*20)
check_cal_result_done(process_str_list,"BaderCharge",10)

print("*"*20+"3.Making data.save and read bader charge infos..."+"*"*20)
if not os.path.exists("data.save"):
    os.system("mkdir data.save")
batch_command(process_str_list,f"{script_dir}/read_bader.py",f"data.save/bader_charge_of_*.data")

print("*"*20+"4.Extracting structures and Read surrounding infos..."+"*"*20)
batch_command(process_str_list,f"{script_dir}/qe_out_reader_gnn.py",f"*/out_relax_*.xsf","xsf -n")
batch_command(process_str_list,f"{script_dir}/get_surrounding_by_elements_multi.py",f"data.save/SURROUNDING_ATOMS_of_*.txt")

print("*"*20+"5.Statistic surrounding infos..."+"*"*20)
batch_command(process_str_list,f"{script_dir}/separate_sur_atom_info_lst.py",f"data.save/Separate_Surrouding_List_of_*.txt")
batch_command(process_str_list,f"{script_dir}/separate_sur_atom_info.py",f"data.save/Separate_Surrouding_of_*.txt")

print("*"*20+"6.Calculating octahedral infos..."+"*"*20)
batch_command(process_str_list,f"{script_dir}/atomic_info_extractor.py",f"data.save/ATOMIC_INFO_of_*.txt")

print("*"*20+"Final: Exracting typical clustering exchange structure..."+"*"*20)
batch_command(process_str_list,f"{script_dir}/create_exchange_str.py","lanchoxie.phd")

print("#"*20+"Creating loop_schedule.py Process On..."+"#"*20+"\n")
print("*"*20+"1.Modifying in_scf,FLAT_INFO and copy necessary files into exchange_dir..."+"*"*20)

print("in_relax","calculation","relax","&control","    ")
print(f"modify: {script_dir}/flat.save/","FLAT_INFO","ppn_num","16","node_num","")
print(f"modify: {script_dir}/flat.save/","FLAT_INFO","wall_time","\"10:00:00\"","ppn_num","")
modify_flat(f".","in_relax","calculation","\"relax\"","&control","    ")
modify_flat(f"{script_dir}/flat.save","FLAT_INFO","ppn_num","16","node_num","")
modify_flat(f"{script_dir}/flat.save","FLAT_INFO","wall_time","\"10:00:00\"","ppn_num","")

print("*"*20+"2.Creating loop_schedule.py in exchange_dir..."+"*"*20)
os.system(f"cp {script_dir}/create_loop_schedule.py .")
os.system(f"cp {script_dir}/loop_schedule.py .")
os.system(f"python create_loop_schedule.py exchange_dir relax")
