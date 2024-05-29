import os
import sys
import numpy as np

str_name_tot=sys.argv[1]
str_name=str_name_tot.split("/")[-1].strip()
str_dir=str_name_tot.split(str_name)[0][:-1] if "/" in str_name_tot else str_name

def split_lines(infiles,split_syb=None,split_mode=None):
    #f=open(infiles,"r+")
    #f1=f.readlines()
    if split_syb==None:
        split_syb=' '
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
                elif split_mode=="int":
                    read_data_row.append(int(i.strip()))
        read_data.append(read_data_row)
    return read_data

raw_data=open(str_name_tot).readlines()
lattice=split_lines(raw_data[2:5],split_mode='float')
element=split_lines([raw_data[5]],split_mode='str')
element_count=split_lines([raw_data[6]],split_mode='int')
file_style=raw_data[7].strip("\n").strip().lower()
coords=split_lines(raw_data[8:],split_mode='float')
element_list=[]
for i,v in enumerate(element[0]):
    element_list.extend([v]*element_count[0][i])

if file_style=='direct':
    lattice_matrix=np.array(lattice)
    coords_car=[]
    # 转换每个分数坐标
    for coord in coords:
        # 将分数坐标转换为 Numpy 数组
        frac_coords_array = np.array(coord[:3])
        # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
        cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
        # 将笛卡尔坐标添加到结果列表
        coords_car.append(cart_coords_array.tolist())
elif file_style=='cartesian':
    coords_car=coords.copy()
else:
    raise ValueError("Unknow formation!")

output=[]
output.append("DIM-GROUP\n")
output.append("           3           1\n")
output.append(" PRIMVEC\n")
for i in lattice:
    output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
output.append(" PRIMCOORD\n")
output.append(f"          {len(coords)}           1\n")
for i,v in enumerate(coords_car):
    output.append(f" {element_list[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 
outname=str_name.replace(".vasp",".xsf")
f1=open(outname,"w+")
for i in output:
    f1.writelines(i)    
print(f"{outname} created!")
