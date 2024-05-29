import numpy as np
import os
import math
import sys
import re

#!!!Notice when using plus:
#The duplicated one will be only calculated once!

#'atom_name', 'bader_charge', 'lowdin_charge'
#(Ni2:100-108)     	136.38840000	151.05865200	
#(91-99)   	126.78930000	141.66762900	
#(Ni2,91-99)	263.17770000	292.72628100	
#(Ni2:100-108)     	136.38840000	151.05865200	
#(91-104)  	202.55530000	225.59025600	
#(Ni2,91-104)	263.17770000	292.72628100	


str_name=sys.argv[1]
input_str=sys.argv[2]   #{Ni,1-2,3-7,8};Ni or (Ni,Li) or Ni   such strings 
input_properties=sys.argv[3]  #searched properties

data_dir="data.save"
data_file=f"{data_dir}/bader_charge_of_{str_name}.data"
mag_file=f"{data_dir}/Mag_of_{str_name}.txt"
lowdin_file=f"{data_dir}/Lowdin_of_{str_name}.txt"
surr_file=f"{data_dir}/SURROUNDING_ATOMS_of_{str_name}.txt"
surr_static_file=f"{data_dir}/Separate_Surrouding_List_of_{str_name}.txt"
surr_number_file=f"{data_dir}/Separate_Surrouding_of_{str_name}.txt"
atomic_info_file=f"{data_dir}/ATOMIC_INFO_of_{str_name}.txt"
data_read_orientation=[['atom_name',0,data_file],   #[properties_name,column_in_files(start from 0),files]
               ['bader_charge',2,data_file],
               ['lowdin_charge',1,data_file],
               ['bader_charge_change',4,data_file],
               ['lowdin_charge_change',3,data_file],
               ['magnetic',-1,mag_file],
               ['tot_occ_lowdin_No_scaled',1,lowdin_file],
               ['s_occ_lowdin_No_scaled',2,lowdin_file],
               ['p_occ_lowdin_No_scaled',3,lowdin_file],
               ['d_occ_lowdin_No_scaled',4,lowdin_file],
               ['atom_chemical_envs',0,surr_file],
               ['surrounding_Li_envs',1,surr_file],
               ['surrounding_TM_envs',2,surr_file],
               ['surrounding_O_envs',3,surr_file],
]

############################################################################################################################
def exactrator_prop(file_in):
    if not os.path.isfile(file_in):
        return
    f_prop=open(file_in).readline()
    props_lst_ex=[]
    props_lst_ex=[x for x in f_prop.strip("#").strip("\n").split() if len(x)>0]
    for i,v in enumerate(props_lst_ex):
        if v not in [x[0] for x in data_read_orientation]:
            data_read_orientation.append([v,i,file_in])

exactrator_prop(surr_static_file)
exactrator_prop(surr_number_file)
exactrator_prop(atomic_info_file)



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
print("#NOT_FOR_CSV# ",end='')
for i in ele_type:
    print(i,end=',')
print()
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
                                data_buffer=n[0]
                                atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    else: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    data_buffer=int(m)
                    atomic_buffer_seg.append(data_buffer)
            elif "-" in m:
                if (int(m.split("-")[0]) >= int(m.split("-")[1])):
                    err_min=int(m.split("-")[0])
                    err_max=int(m.split("-")[1])
                    raise ValueError(f"{err_min} not less than {err_max}!")
                for n in range(int(m.split("-")[0]),int(m.split("-")[1])+1):
                    data_buffer=n
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
                                data_buffer=n[0]
                                multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n[1]}_{n[0]})")
                                atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    elif m not in ele_type: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    data_buffer=int(m)
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
                        data_buffer=n
                        multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n})")
                        atomic_buffer_seg.append(data_buffer)
        for i in atomic_buffer_seg:
            atomic_buffer.append([i]) 

    elif "(" not in v and ")" not in v and "{" not in v and "}" not in v:
        if v.split("-")[0] in ele_type:
            for j,m in enumerate(file_dic):
                if m[1]==v.split("-")[0]:
                    data_buffer=m[0]
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


#for i,v in enumerate(input_seg):
#    print(v,atomic_buffer[i])

#print(atomic_buffer)

#print(properties_lst)

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
    i_tail=i.copy()
    tails_buffer=''
    #for j in i_tail:
    #    tails_buffer+="-"+str(j)
    number_buffer=sorted(i)
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
print("#NOT_FOR_CSV#",detected_dic)


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
#detected_dic,atomic_dic_files,draw_index,read_lines_from_pdos_lst is available in below section
def read_files_str(infiles):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if lines[0]=="#":
            continue
        if "Direct" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split()
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(str(i))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data

data_dic={}
for i,v in enumerate(data_read_orientation):
    if os.path.isfile(v[2]):
        data_dic[v[0]]=[x[v[1]] for x in read_files_str(v[2])]
print("#AV_PROP#","\t".join([properties for properties,values in data_dic.items()]))
print("#TOT_PROP#","\t".join([x[0] for x in data_read_orientation]))
#data_dic['atom_name']=[x[0] for x in get_datas]
#data_dic['bader_charge']=[x[1] for x in get_datas]
#data_dic['lowdin_charge']=[x[2] for x in get_datas]
#data_dic['bader_charge_change']=[x[3] for x in get_datas]
#data_dic['lowdin_charge_change']=[x[4] for x in get_datas]
#data_dic['magnetic']=[x[5] for x in get_datas]

#judge avalaible properties
props_default=[]
properties_lst=[x for x in input_properties.strip().split("+") if len(x)>0]
for i in properties_lst:
    if i not in [properties for properties,values in data_dic.items()]:
        if (i[-1]==("*"))&(i[:-1] not in [properties for properties,values in data_dic.items()]):
            props_default.append(i)
        elif i[:-1] in [properties for properties,values in data_dic.items()]:
            properties_lst[properties_lst.index(i)]=i[:-1]
        else:
            raise ValueError(f"{i} not found in {[properties for properties,values in data_dic.items()]}!")
output_data={}
output_data['atom_name']=[]
for i in properties_lst:
    if i not in props_default:
        output_data[i]=[]
    else:
        output_data[i[:-1]]=[]
#print(properties_lst)
#print(output_data)
#print("****")
#print(atomic_buffer)
#print(props_default)
for i,v in enumerate(atomic_buffer):
    if len(v)==1:
        if v[0]==int(data_dic['atom_name'][v[0]-1].split("-")[-1]):
            output_data['atom_name'].append(data_dic['atom_name'][v[0]-1])
        else:
            raise ValueError(f"{v[0]} did not match with {output_data['atom_name'][v[0]-1]} !")
    else:
        input_str="".join([x for x in input_seg[i] if not x.isdigit() and x!=","])
        #print(input_str,len(input_str))
        if len(input_str)>2:
            output_data['atom_name'].append(input_seg[i])
        else:
            #print("Con")
            number_buffer=v.copy()
            number_buffer=sorted(number_buffer)
            output_data['atom_name'].append(f"({convert_to_ranges(number_buffer)})")
    for prop in properties_lst:
        if (prop in props_default)&(prop not in [properties for properties,values in data_dic.items()]):
            output_data[prop[:-1]].append(0)
        elif (prop!='atom_name'):
            try:
                int(data_dic[prop][v[0]-1])
            except:
                try:
                    float(data_dic[prop][v[0]-1])
                except:
                    if len(v)>1:
                        data_i="No_sum_data"
                    elif len(v)==1:
                        data_i=data_dic[prop][v[0]-1]
                else:
                    data_i=0
                    for j in v:
                        data_i+=float(data_dic[prop][j-1])
            else:
                data_i=0
                for j in v:
                    data_i+=float(data_dic[prop][j-1])
            output_data[prop].append(data_i)
print("\t".join([properties for properties,values in output_data.items()]))
for i in range(len(atomic_buffer)):
    for prop in [properties for properties,values in output_data.items()]:
        try:
            float(output_data[prop][i])
        except:
            print("%-10s\t"%output_data[prop][i],end='')
        else:
            if (output_data[prop][i]%1!=0):
                print("%-10.6f\t"%float(output_data[prop][i]),end='')
            else:
                print("%-10d\t"%float(output_data[prop][i]),end='')
    print()
