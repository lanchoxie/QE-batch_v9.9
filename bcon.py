import os
import sys
from collections import Counter
import numpy as np
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Read QE relax/vc-relax output and convert specific step into xsf/vasp/qein format or all step into xsf format")

# 添加参数
parser.add_argument('out_name', type=str, help='output filename')
parser.add_argument('in_name', type=str, help='input filename')
# 解析命令行参数
args = parser.parse_args()

input_file=args.in_name
output_file=args.out_name
specific_step=-1

if not os.path.isfile(output_file):
    print(f"No {output_file},exit")
    sys.exit(0)
if not os.path.isfile(input_file):
    print(f"No {input_file},exit")
    sys.exit(0)

input_cont=open(input_file).readlines()
atom_number=0
for line in input_cont:
    if "nat" in line and "!" not in line:
        atom_number=int(line.strip("\n").split(",")[0].split("=")[1].strip())
if atom_number==0:
    raise ValueError("No nat found in inputfile!")
output_cont=os.popen(f"grep 'ATOMIC' -A {atom_number} {output_file}").readlines()
#for i in output_cont:
#    print(i)

lattice_raw=os.popen(f"grep CELL_PARAMETERS -A 3 {input_file}").readlines()
lattice_vc_raw=os.popen(f"grep CELL_PARAMETERS -A 3 {output_file}").readlines()
if len(lattice_vc_raw)==0:
    mode="relax"
else:
    mode="vc-relax"
total_str_count=len([x for x in output_cont if x.find("ATOMIC")!=-1])

print(mode,f"{total_str_count} step")

if total_str_count==0:
    print("No relaxation found, exit!")
    sys.exit(0)
#for i in lattice:
#    print(i)
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
                    read_data_row.append(str(i))
                elif split_mode=="float":
                    read_data_row.append(float(i))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    return read_data

def parse_cell_parameters(input_lines,tag_line,skip=0):
    lattice_vectors = []  # 用于存储所有的晶格向量
    current_matrix = []  # 临时存储当前正在解析的晶格矩阵

    for line in input_lines:
        # 去除行尾的换行符和空白字符
        line = line.strip()
        # 跳过空行和'--'分隔符
        if line == '' or line == '--':
            # 如果当前矩阵不为空，说明我们已经完成了一个晶格的解析
            if current_matrix:
                lattice_vectors.append(current_matrix)
                current_matrix = []  # 重置当前矩阵，为解析下一个晶格做准备
            continue  # 继续处理下一行

        # 分析晶格参数行
        if line.find(tag_line)!=-1:
            continue  # 跳过包含CELL_PARAMETERS的行
        else:
            # 解析包含晶格向量的行
            vector = [float(x) for x in line.split()[skip:]]
            current_matrix.append(vector)

    # 添加最后一个晶格矩阵（如果有的话）
    if current_matrix:
        lattice_vectors.append(current_matrix)

    return lattice_vectors


lattice_vectors_vc = parse_cell_parameters(lattice_vc_raw,"CELL_PARAMETERS")
str_vectors = parse_cell_parameters(output_cont,"ATOMIC_POSITIONS",skip=1)

if len(str_vectors[-1])<atom_number:
    print("Last output corrupted, exit!")
    sys.exit(0)
atom_species=[i[0] for i in split_lines(output_cont[1:atom_number+1]," ",split_mode="str")]    #储存所有原子的序号（重复）
#print(len(atom_species),atom_species)
lattice_vec=split_lines(lattice_raw[1:]," ",split_mode="float")


def create_out_file(lat,coords,species):
    output=[]
    lat_tag_line=""
    coord_tag_line=""
    read_syb=0
    for i,v in enumerate(input_cont):
        if "ATOMIC_POSITIONS" in v:
            coord_tag_line=v
        if "CELL_PARAMETERS" in v:
            lat_tag_line=v
            read_syb=i
    for i in input_cont[:read_syb]:
        output.append(i)
    output.append(lat_tag_line)
    for i in lat:
        output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
    output.append(coord_tag_line)
    for i,v in enumerate(coords):
        if len(v)==3:
            v.extend([1,1,1])
        #print(v)
        output.append(f" {species[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f} {int(v[3])} {int(v[4])} {int(v[5])}\n") 
    return output            

def create_output(out,name):
    count = sum(1 for f in os.listdir('.') if output_file in f)
    os.system(f"cp {input_file} {input_file}.{count}")
    os.system(f"cp {output_file} {output_file}.{count}")
    f_out=open(f"{name}","w+")
    for i in out:
        f_out.writelines(i)
    f_out.close()
    print(f"{name} Created!")
    

if mode=="relax":
    output_=create_out_file(lattice_vec,str_vectors[specific_step],atom_species)
elif mode=="vc-relax":
    output_=create_out_file(lattice_vectors_vc[specific_step],str_vectors[specific_step],atom_species)

out_filename=input_file

create_output(output_,out_filename)
#os.system(f"mv {out_filename} {str_name}")
