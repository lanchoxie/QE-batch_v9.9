import os
import sys
from collections import Counter
import numpy as np
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="Read QE relax/vc-relax output and convert specific step into xsf/vasp/qein format or all step into xsf format")

# 添加参数
parser.add_argument('str_name', type=str, help='input filename')
parser.add_argument('output_type', type=str, help='xsf/xsf_std/vasp/qein, where xsf output element and xsf_std output atomic weight')
parser.add_argument('--neate', '-n', action='store_true', help='if add, then did not create file in current directory and only create in target directory')
# 解析命令行参数
args = parser.parse_args()

str_name=args.str_name.split("/")[-1].strip()
#str_dir=args.str_name.split(str_name)[0][:-1] if "/" in args.str_name else str_name
str_dir=args.str_name
input_file=f"{str_dir}/in_relax_{str_name}"
output_file=f"{str_dir}/out_relax_{str_name}"
output_type=args.output_type
specific_step=-1

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

def rl_aw(element_i):
    element=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Uun","Uuu","Uub"]   #end in 112
    #return element[int(element_i-1)]
    return element.index(element_i)+1

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

def elements_and_counts(elements):
    # Use Counter to get the counts of each element
    counts = Counter(elements)
    # Separate the elements and their counts into two lists
    elements_list = list(counts.keys())
    counts_list = list(counts.values())
    return elements_list, counts_list



lattice_vectors_vc = parse_cell_parameters(lattice_vc_raw,"CELL_PARAMETERS")
str_vectors = parse_cell_parameters(output_cont,"ATOMIC_POSITIONS",skip=1)

#atom_species_wt=[rl_aw(i[0]) for i in split_lines(output_cont[1:atom_number+1]," ")]    #储存所有原子的序号（重复）
atom_species=[i[0] for i in split_lines(output_cont[1:atom_number+1]," ",split_mode="str")]    #储存所有原子的序号（重复）
#print(len(atom_species),atom_species)
lattice_vec=split_lines(lattice_raw[1:]," ",split_mode="float")


def create_out_file(out_type,lat,coords,species):
    output=[]

    if out_type=="xsf_std":
        lattice_matrix=np.array(lat)
        coords_car=[]
        # 转换每个分数坐标
        for coord in coords:
            # 将分数坐标转换为 Numpy 数组
            frac_coords_array = np.array(coord[:3])
            # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
            cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
            # 将笛卡尔坐标添加到结果列表
            coords_car.append(cart_coords_array.tolist())
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" CONVVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords_car):
            output.append(f" {rl_aw(species[i])}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="xsf":
        lattice_matrix=np.array(lat)
        coords_car=[]
        # 转换每个分数坐标
        for coord in coords:
            # 将分数坐标转换为 Numpy 数组
            frac_coords_array = np.array(coord[:3])
            # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
            cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
            # 将笛卡尔坐标添加到结果列表
            coords_car.append(cart_coords_array.tolist())
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords_car):
            output.append(f" {species[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="vasp":
        elements, counts = elements_and_counts(species)
        output.append(f"VASP FILE {input_file} {output_file} STEP {specific_step}\n")
        output.append("1.0\n")
        for i in lat:
            print(i)
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" "+"  ".join(elements)+"\n")
        output.append(" "+"  ".join([str(x) for x in counts])+"\n")
        output.append("Direct\n")
        for i,v in enumerate(coords):
            output.append(f"   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="qein":
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
        
    elif out_type=="all":
        output.append(f"ANIMSTEPS {total_str_count}\n")
        output.append(f"CRYSTAL\n")
        for j,coord_i in enumerate(coords):
            lattice_matrix=np.array(lat[j])
            coords_car=[]
            # 转换每个分数坐标
            for coord in coord_i:
                # 将分数坐标转换为 Numpy 数组
                frac_coords_array = np.array(coord)
                # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
                cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
                # 将笛卡尔坐标添加到结果列表
                coords_car.append(cart_coords_array.tolist())
            output.append(f" PRIMVEC {j}\n")
            for i in lat[j]:
                output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
            output.append(f" CONVVEC {j}\n")
            for i in lat[j]:
                output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
            output.append(" PRIMCOORD {j}\n")
            output.append(f"          {len(coord_i)}           1\n")
            for i,v in enumerate(coords_car):
                output.append(f" {rl_aw(species[i])}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 
    return output            

def create_output(out,name):
    f_out=open(f"{str_dir}/{name}","w+")
    for i in out:
        f_out.writelines(i)
    f_out.close()
    if not args.neate:
        os.system(f"cp {str_dir}/{name} .")
        print(f"{name} Created in current dir!")
    print(f"{str_dir}/{name} Created!")
    

if mode=="relax":
    output_=create_out_file(output_type,lattice_vec,str_vectors[specific_step],atom_species)
elif mode=="vc-relax":
    output_=create_out_file(output_type,lattice_vectors_vc[specific_step],str_vectors[specific_step],atom_species)

if output_type=="xsf":
    out_filename=f"out_relax_{str_name}.xsf"
elif output_type=="xsf_std":
    out_filename=f"out_relax_{str_name}_standard.xsf"
elif output_type=="vasp":
    out_filename=f"out_relax_{str_name}.vasp"
elif output_type=="qein":
    out_filename=f"buffer_in_relax_{str_name}"

create_output(output_,out_filename)
#os.system(f"mv {out_filename} {str_name}")
