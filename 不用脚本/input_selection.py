import numpy as np
import os
import math
import sys
import re

str_name=sys.argv[1]
input_str=sys.argv[2]   #{Ni,1-2,3-7,8};Ni or (Ni,Li) or Ni   such strings 
input_properties=sys.argv[3]  #searched properties

data_file=""
############################################################################################################################
file_dic=[]
if os.path.isfile(f"{str_name}/in_scf_{str_name}"):
    input_file_content=open(f"{str_name}/in_scf_{str_name}").readlines()
elif os.path.isfile(f"{str_name}/in_relax_{str_name}"):
    input_file_content=open(f"{str_name}/in_relax_{str_name}").readlines()

atom_num=0
for lines in input_file_content:
    if "nat" in lines and "!" not in lines:
        atom_num=int(lines.split("=")[-1].strip().strip("\n").strip(",").strip())
if atom_num==0:
    raise ValueError("\'nat\' not found in input file!!")

read_start=0
read_count=1
atomic_species=[]
for i,v in enumerate(input_file_content):
    if 'ATOMIC_POSITIONS' in v:
        read_start=1
        continue
    if (read_start==1)&(read_count<atom_num+1):
        read_count+=1
        atomic_species.append([x for x in v.split() if len(x)>0][0]) 
ele_type=list(set(atomic_species))
for i,v in enumerate(atomic_species): 
    file_dic.append([i+1,v])

input_seg=[i.strip() for i in input_str.strip().split(";") if len(i)!=0]#decode the input into [atomic_number,orbital_number,orbital_type]
def string_match(str_in):
    #pattern1 = r'^[(][^;]+[)]-(?:\d+-[A-Za-z_0-9]+|tot)$'
    #pattern2 = r'^[{][^;]+[}]-(?:\d+-[A-Za-z_0-9]+|tot)$'
    #pattern3 = r'^[A-Za-z]+-(?:\d+-[A-Za-z_0-9]+|tot)$'
    #patterns=[pattern1,pattern2,pattern3]
    pattern1 = r'^[(][^;]+[)]$'
    pattern2 = r'^[{][^;]+[}]$'
    pattern3 = r'^[A-Za-z0-9]+' #0-9 here to match the spin up and spin down element
    patterns=[pattern1,pattern2,pattern3]
    matched=0
    for pattern in patterns:
        if re.match(pattern, str_in):
            matched=1
            return True
    if matched==0:
        raise ValueError("111the segment %s is invalid! plz type in {/(number-number,element,...}/)"%str_in)

for i in input_seg:
    string_match(i)

atomic_buffer=[]
multi_split=[]
for i,v in enumerate(input_seg):
    atomic_buffer_seg=[]
    if "(" in v and ")" in v:
        numb_seg=[s.strip() for s in v.split(")")[0].split("(")[1].split(",")]
        for j,m in enumerate(numb_seg):
            if "-" not in m:
                try:
                    int(m)
                except:  # element in the {}
                    if m in ele_type:
                        for k,n in enumerate(file_dic):
                            if n[1]==m:
                                data_buffer=[n[0]]
                                if data_buffer not in atomic_buffer_seg:
                                    atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    else: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    data_buffer=[int(m)]
                    if data_buffer not in atomic_buffer_seg:
                        atomic_buffer_seg.append(data_buffer)
            elif "-" in m:
                if (int(m.split("-")[0]) >= int(m.split("-")[1])):
                    err_min=int(m.split("-")[0])
                    err_max=int(m.split("-")[1])
                    raise ValueError(f"{err_min} not less than {err_max}!")
                for n in range(int(m.split("-")[0]),int(m.split("-")[1])+1):
                    data_buffer=[n]
                    if data_buffer not in atomic_buffer_seg:
                        atomic_buffer_seg.append(data_buffer)
        atomic_buffer.append(atomic_buffer_seg)
    elif "{" in v and "}" in v:
        multi_split.append([i,[]])
        numb_seg=[s.strip() for s in v.split("}")[0].split("{")[1].split(",")]
        for j,m in enumerate(numb_seg):
            if "-" not in m:
                try:
                    int(m)
                except:  # element in the {}
                    if m in ele_type:
                        for k,n in enumerate(file_dic):
                            if n[1]==m:
                                data_buffer=[n[0]]
                                if data_buffer not in atomic_buffer_seg:
                                    multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n[1]}_{n[0]})")
                                    atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    elif m not in ele_type: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    data_buffer=[int(m)]
                    if data_buffer not in atomic_buffer_seg:
                        multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({m})")
                        atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer_seg.append([int(m),int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]])
            elif "-" in m:
                if (int(m.split("-")[0]) >= int(m.split("-")[1])):
                    err_min=int(m.split("-")[0])
                    err_max=int(m.split("-")[1])
                    raise ValueError(f"{err_min} not less than {err_max}!")
                for n in range(int(m.split("-")[0]),int(m.split("-")[1])+1):
                    if 'tot' not in v:
                        data_buffer=[n]
                        if data_buffer not in atomic_buffer_seg:
                            multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n})")
                            atomic_buffer_seg.append(data_buffer)
        for i in atomic_buffer_seg:
            atomic_buffer.append([i]) 

    elif "(" not in v and ")" not in v and "{" not in v and "}" not in v:
        if v.split("-")[0] in ele_type:
            for j,m in enumerate(file_dic):
                if m[1]==v.split("-")[0]:
                    data_buffer=[m[0]]
                    if data_buffer not in atomic_buffer_seg:
                        atomic_buffer_seg.append([m[0]])
            atomic_buffer.append(atomic_buffer_seg)
        else:
            err_ele=v.split("-")[0]
            raise ValueError(f"{err_ele} not in {ele_type}")
    else:
        raise ValueError("111the segment %s is invalid! plz type in {/(number-number,element,...}/)"%str_in)

for i in input_seg[::-1]:
    if input_seg.index(i) in [x[0] for x in multi_split]:
        for j in multi_split[[x[0] for x in multi_split].index(input_seg.index(i))][1]:
            input_seg.insert(input_seg.index(i),j)
        input_seg.remove(i)


for i,v in enumerate(input_seg):
    print(v,atomic_buffer[i])

#print(atomic_buffer)

properties_lst=[x for x in input_properties.strip().split("-") if len(x)>0]
print(properties_lst)

def convert_to_ranges(lst):
    ranges = []
    i = 0
    while i < len(lst):
        start = lst[i]
        while i + 1 < len(lst) and lst[i + 1] - lst[i] == 1:
            i += 1
        end = lst[i]
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        i += 1
    return ",".join(ranges)
def find_duplicates(lst): #find the duplicates element
    duplicates = []
    seen = set()
    for i, x in enumerate(lst):
        if x in seen:
            duplicates.append([i,lst.index(x)])
        else:
            seen.add(x)
    return duplicates
    #return [i for i, x in enumerate(lst) if lst.count(x) > 1]

detected_file=[]
for i in atomic_buffer:
    i_tail=i[0].copy()
    i_tail.remove(i_tail[0])
    tails_buffer=''
    for j in i_tail:
        tails_buffer+="-"+str(j)
    number_buffer=[x[0] for x in i]
    number_buffer=sorted(number_buffer)
    #print(number_buffer)
    out_words=f"({convert_to_ranges(number_buffer)}){tails_buffer}"
    #print(out_words)
    detected_file.append(out_words)

#print(detected_file)
duplicate_output=find_duplicates(detected_file)
#print(duplicate_output,len(duplicate_output))
#for i in duplicate_output:
    #print(detected_file[i[0]],input_seg[i[0]],"same as:",input_seg[i[1]],)
#print([x[0] for x in duplicate_output])

atomic_dic=[]
detected_dic=[]
draw_index=[]
read_lines_from_pdos_lst=[]
for i,v in enumerate(detected_file):
    # IN a second thought I think there is no need for remove the duplicated element cuz the following program has prevent the repeat calculation
    #if i not in [x[0] for x in duplicate_output]:
        #atomic_dic.append(atomic_buffer[i])
        #detected_dic.append(detected_file[i])
    atomic_dic.append(atomic_buffer[i])
    detected_dic.append(detected_file[i])
print(detected_dic)


#detected_dic,atomic_dic_files,draw_index,read_lines_from_pdos_lst is available in below section

