import numpy as np
import os
import math
import sys
import re

#This script only works in the Calculation file!!
#This script did not support element with f orbitals valance electrons

#for p orbital,it is  pz px py
#for d orbital,it is  dz2 dzx dzy dx2-y2 dxy

result_file="PDOS_DIY_RESULT"
output_file_head='PDOS_OF_INDEX_'
if os.path.isfile(result_file):
    result_his_line=open(result_file).readlines()
    result_his_lst=[]
    for i in result_his_line:
        result_his_lst.append([i.strip("\n").split("\t")[0],i.strip("\n").split("\t")[1]])
else:
    result_his_lst=[]

common_str_in_output_file="BTO.pdos_atm#"
input_str=sys.argv[1]

dir_file=(os.popen("pwd").read())
dir_current=max(dir_file.split('\n'))

file_list_tot=[]
file_input_tag=[]
for root,dirs,files in os.walk(dir_current):
    for file in files:
        if "in_scf" in file or "in_relax" in file:
            file_input_tag.append(file)
        if common_str_in_output_file in file:
            file_list_tot.append(file)

if len(file_input_tag)==0:
    raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")

spin_mode=0
input_tag=open(file_input_tag[0]).readlines()
for lines in input_tag:
    if "nspin" in lines and "2" in lines and "!" not in lines:
        spin_mode=1   

file_dic=[]
for i,files in enumerate(file_list_tot): #read all the pdos files in the dir and store them in [atomic_number,atomic_element,orbital_number,orbital_type]
    atomic_nb_i=int(files.split("#")[1].split("(")[0])
    atomic_ele_i=files.split("#")[1].split("(")[1].split(")")[0]
    orbit_nb_i=int(files.split("#")[-1].split("(")[0])
    orbit_tp_i=files.split("#")[-1].split("(")[1].split(")")[0]
    file_dic.append([atomic_nb_i,atomic_ele_i,orbit_nb_i,orbit_tp_i])
file_dic.sort(key=lambda x:(x[0],x[2]))
orbit_proj=[['s',['s'],[3+spin_mode]],['p',['p','pz','px','py'],[2,3+spin_mode,4+2*spin_mode,5+3*spin_mode]],['d',['d','dz2','dzx','dzy','dx2_y2','dxy'],[2,3+spin_mode,4+2*spin_mode,5+3*spin_mode,6+4*spin_mode,7+5*spin_mode]]]
ele_type=list(set([x[1] for x in file_dic]))
ele_range_buffer=[]
for i,v in enumerate(file_dic):
    if v[1] not in [x[0] for x in ele_range_buffer]:
        ele_range_buffer.append([v[1],[v[0]]])
    elif v[1] in [x[0] for x in ele_range_buffer]:
        if v[0] not in ele_range_buffer[[x[0] for x in ele_range_buffer].index(v[1])][1]:
            ele_range_buffer[[x[0] for x in ele_range_buffer].index(v[1])][1].append(v[0])
ele_range=[]
for i in ele_range_buffer:
    ele_range.append([i[0],min(i[1]),max(i[1])])
ele_orbit=[]# this list stores the info like [['Ni1', ['-1-s', '-2-p', '-3-s', '-4-d']], ['Li', ['-1-s', '-2-s']], ['O', ['-1-s', '-2-p']], ['Co', ['-1-s', '-2-p', '-3-s', '-4-d']], ['Ni2', ['-1-s', '-2-p', '-3-s', '-4-d']]]
for i,v in enumerate(file_dic):
    v_orb='-'+str(v[2])+'-'+v[3]
    if v[1] not in [x[0] for x in ele_orbit]:
        ele_orbit.append([v[1],[v_orb]])
    elif v[1] in [x[0] for x in ele_orbit]:
        if v_orb not in ele_orbit[[x[0] for x in ele_orbit].index(v[1])][1]:
            ele_orbit[[x[0] for x in ele_orbit].index(v[1])][1].append(v_orb)
#print(ele_type)
###############################THIS IS USEFUL AND SHALL BE PRINT OUT IN APP_*.py############################################
print(ele_range)
print(ele_orbit)
############################################################################################################################


input_seg=[i.strip() for i in input_str.strip().split(";") if len(i)!=0]#decode the input into [atomic_number,orbital_number,orbital_type]
def string_match(str_in):
    pattern1 = r'^[(][^;]+[)]-(?:\d+-[A-Za-z_0-9]+|tot)$'
    pattern2 = r'^[{][^;]+[}]-(?:\d+-[A-Za-z_0-9]+|tot)$'
    pattern3 = r'^[A-Za-z]+-(?:\d+-[A-Za-z_0-9]+|tot)$'
    patterns=[pattern1,pattern2,pattern3]
    matched=0
    for pattern in patterns:
        if re.match(pattern, str_in):
            matched=1
            return True
    if matched==0:
        raise ValueError("111the segment %s is invalid! plz type in {/(number-number,element,...}/)-oribit_number-orbital [orbit_number-orbital segment could be \"tot\"]"%str_in)

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
                                if 'tot' in v:
                                    if [n[0],'tot'] not in atomic_buffer_seg:
                                        atomic_buffer_seg.append([n[0],'tot'])
                                elif 'tot' not in v:
                                    data_buffer=[n[0],int(v.split(")")[1].split("-")[1]),v.split(")")[1].split("-")[2]]
                                    if data_buffer not in atomic_buffer_seg:
                                        atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    else: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    if 'tot' not in v:
                        data_buffer=[int(m),int(v.split(")")[1].split("-")[1]),v.split(")")[1].split("-")[2]]
                        if data_buffer not in atomic_buffer_seg:
                            atomic_buffer_seg.append([int(m),int(v.split(")")[1].split("-")[1]),v.split(")")[1].split("-")[2]])
                    elif 'tot' in v:
                        if [int(m),'tot'] not in atomic_buffer_seg:
                            atomic_buffer_seg.append([int(m),'tot'])
            elif "-" in m:
                if (int(m.split("-")[0]) >= int(m.split("-")[1])):
                    err_min=int(m.split("-")[0])
                    err_max=int(m.split("-")[1])
                    raise ValueError(f"{err_min} not less than {err_max}!")
                for n in range(int(m.split("-")[0]),int(m.split("-")[1])+1):
                    if 'tot' not in v:
                        data_buffer=[n,int(v.split(")")[1].split("-")[1]),v.split(")")[1].split("-")[2]]
                        if data_buffer not in atomic_buffer_seg:
                            atomic_buffer_seg.append([n,int(v.split(")")[1].split("-")[1]),v.split(")")[1].split("-")[2]])
                    elif 'tot' in v:
                        if [n,'tot'] not in atomic_buffer_seg:
                            atomic_buffer_seg.append([n,'tot'])
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
                                if 'tot' in v:
                                    if [n[0],'tot'] not in atomic_buffer_seg:
                                        multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n[1]}_{n[0]})-'tot'")
                                        atomic_buffer_seg.append([n[0],'tot'])
                                elif 'tot' not in v:
                                    data_buffer=[n[0],int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]]
                                    if data_buffer not in atomic_buffer_seg:
                                        multi_ob_nm=v.split("}")[1].split("-")
                                        multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n[1]}_{n[0]})-{multi_ob_nm[1]}-{multi_ob_nm[2]}")
                                        atomic_buffer_seg.append(data_buffer)
                        #atomic_buffer.append(atomic_buffer_seg)
                    elif m not in ele_type: # input element not in list
                        err_ele=m
                        raise ValueError(f"{err_ele} not in {ele_type}")
                else: #atomic number in the {}
                    if 'tot' not in v:
                        data_buffer=[int(m),int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]]
                        if data_buffer not in atomic_buffer_seg:
                            multi_ob_nm=v.split("}")[1].split("-")
                            multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({m})-{multi_ob_nm[1]}-{multi_ob_nm[2]}")
                            atomic_buffer_seg.append(data_buffer)
                            #atomic_buffer_seg.append([int(m),int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]])
                    elif 'tot' in v:
                        if [int(m),'tot'] not in atomic_buffer_seg:
                            multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({m})-'tot'")
                            atomic_buffer_seg.append([int(m),'tot'])
            elif "-" in m:
                if (int(m.split("-")[0]) >= int(m.split("-")[1])):
                    err_min=int(m.split("-")[0])
                    err_max=int(m.split("-")[1])
                    raise ValueError(f"{err_min} not less than {err_max}!")
                for n in range(int(m.split("-")[0]),int(m.split("-")[1])+1):
                    if 'tot' not in v:
                        data_buffer=[n,int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]]
                        if data_buffer not in atomic_buffer_seg:
                            multi_ob_nm=v.split("}")[1].split("-")
                            multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n})-{multi_ob_nm[1]}-{multi_ob_nm[2]}")
                            atomic_buffer_seg.append(data_buffer)
                            #atomic_buffer_seg.append([n,int(v.split("}")[1].split("-")[1]),v.split("}")[1].split("-")[2]])
                    elif 'tot' in v:
                        if [n,'tot'] not in atomic_buffer_seg:
                            multi_split[[x[0] for x in multi_split].index(i)][1].append(f"({n})-'tot'")
                            atomic_buffer_seg.append([n,'tot'])
        for i in atomic_buffer_seg:
            atomic_buffer.append([i]) 

    elif "(" not in v and ")" not in v and "{" not in v and "}" not in v:
        if v.split("-")[0] in ele_type:
            for j,m in enumerate(file_dic):
                if m[1]==v.split("-")[0]:
                    if 'tot' in v:
                        if [m[0],'tot'] not in atomic_buffer_seg:
                            atomic_buffer_seg.append([m[0],'tot'])
                    elif 'tot' not in v:
                        data_buffer=[m[0],int(v.split("-")[1]),v.split("-")[2]]
                        if data_buffer not in atomic_buffer_seg:
                            atomic_buffer_seg.append([m[0],int(v.split("-")[1]),v.split("-")[2]])
            atomic_buffer.append(atomic_buffer_seg)
        else:
            err_ele=v.split("-")[0]
            raise ValueError(f"{err_ele} not in {ele_type}")
    else:
        raise ValueError("111the segment %s is invalid! plz type in {/(number-number,element,...}/)-oribit_number-orbital [orbit_number-orbital segment could be \"tot\"]"%v)
        #raise ValueError(f"the segment {v} is invalid! plz type in {/(number-number,element,...}/)-oribit/element-orbit [orbit could be \"tot\"]")

for i in input_seg[::-1]:
    if input_seg.index(i) in [x[0] for x in multi_split]:
        for j in multi_split[[x[0] for x in multi_split].index(input_seg.index(i))][1]:
            input_seg.insert(input_seg.index(i),j)
        input_seg.remove(i)

#print(atomic_buffer)

for j in atomic_buffer:
    for i,v in enumerate(j):
        if v[0] not in [x[0] for x in file_dic]:
            raise ValueError(f"Error: The atomic number is ranging from {min([x[0] for x in file_dic])} to {max([x[0] for x in file_dic])}!")
        if 'tot' in v:
            pass
        elif 'tot' not in v:
            orbit_found=0
            for j in file_dic:
                if (v[1]==j[2])&(v[2][0]==j[3])&(v[0]==j[0]):
                    if v[2] not in orbit_proj[[x[0] for x in orbit_proj].index(v[2][0])][1]:
                        if v[2]==orbit_proj[[x[0] for x in orbit_proj].index(v[2][0])][0]:
                            orbit_found=1
                        else:
                            raise ValueError(f"The available project orbital for {v[2][0]} is {orbit_proj[[x[0] for x in orbit_proj].index(v[2][0])][1]}")
                    else:
                        orbit_found=1
            if orbit_found==0:
                ele_err=file_dic[[x[0] for x in file_dic].index(v[0])][1]
                raise ValueError(f"-{v[1]}-{v[2]} is not found in {ele_err},the available orbit in this psp is:{ele_orbit[[x[0] for x in ele_orbit].index(ele_err)][1]}, and the range of {ele_err} is from {ele_range[[x[0] for x in ele_range].index(ele_err)][1]} to {ele_range[[x[0] for x in ele_range].index(ele_err)][2]}")
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
for i in [x.split("-")[-1] for x in detected_dic]:
    draw_index_i=0
    if i=='tot':
        draw_index_i=2
        read_lines_from_pdos_lst.append(3)
    else:
        orbit_index=[x[0] for x in orbit_proj].index(i[0])
        draw_index_i=orbit_proj[orbit_index][2][orbit_proj[orbit_index][1].index(i)]
        if orbit_proj[orbit_index][0]=='s':
            read_lines_from_pdos_lst.append(3+2*spin_mode)
        if orbit_proj[orbit_index][0]=='p':
            read_lines_from_pdos_lst.append(5+4*spin_mode)
        if orbit_proj[orbit_index][0]=='d':
            read_lines_from_pdos_lst.append(7+6*spin_mode)
    if draw_index_i==0:
        raise ValueError("Something is wrong and I have no idea,seriously :P")
    else: 
        draw_index.append(draw_index_i)
for i,v in enumerate(detected_dic):
    if 'tot' in v:
        pass
    else:
        detected_dic[i]=v.split(")")[0]+")"+"-"+v.split(")")[1].split("-")[1]+"-"+v.split(")")[1].split("-")[2][0]
    print(detected_dic[i],draw_index[i],read_lines_from_pdos_lst[i])

def file_from_num(lst):
    if lst[1]=='tot':
        file_return=[]
        for i,v in enumerate(file_list_tot):
            atomic_nb_in=int(v.split("#")[1].split("(")[0])
            orbit_nb_in=int(v.split("#")[-1].split("(")[0]) 
            if atomic_nb_in==lst[0]:
                file_return.append(v)
        return file_return
    else:
        for i,v in enumerate(file_list_tot):
            atomic_nb_in=int(v.split("#")[1].split("(")[0])
            orbit_nb_in=int(v.split("#")[-1].split("(")[0])
            if (atomic_nb_in==lst[0])&(orbit_nb_in==lst[1]):
                return v

atomic_dic_files=[]
for i,v in enumerate(atomic_dic):
    at_dic_fs_buffer=[]
    for j in v: 
        #print(j,file_from_num(j))
        if j[1]=='tot':
            at_dic_fs_buffer.extend(file_from_num(j))
        if j[1]!='tot':
            at_dic_fs_buffer.append(file_from_num(j))
    atomic_dic_files.append(at_dic_fs_buffer)
#for i,v in enumerate(atomic_dic):
#    print("**********\n")
#    print(detected_dic[i])
#    print(atomic_dic[i])
#    print(atomic_dic_files[i])

#detected_dic,atomic_dic_files,draw_index,read_lines_from_pdos_lst is available in below section

depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if "in_pdos_" in filename:
                str_in_dir.append(filename.split("in_pdos_")[1].split(".")[0])

str_name=str_in_dir[0]

def Etofloat(str_num):
    if 'E' not in str_num:
        if '-' in str_num:
            float_num= -1 * float(str_num)
        elif '-' not in str_num:
            float_num= float(str_num)
    if 'E' in str_num:
        before_e = float(str_num.split('E')[0])
        sign = str_num.split('E')[1][:1]
        after_e = int(str_num.split('E')[1][1:])

        if sign == '+':
            float_num = before_e * math.pow(10, after_e)
        elif sign == '-':
            float_num = before_e * math.pow(10, -after_e)
        else:
            float_num = None
            print('error: unknown sign')
    elif 'e' in str_num:
        before_e = float(str_num.split('e')[0])
        sign = str_num.split('e')[1][:1]
        after_e = int(str_num.split('e')[1][1:])

        if sign == '+':
            float_num = before_e * math.pow(10, after_e)
        elif sign == '-':
            float_num = before_e * math.pow(10, -after_e)
        else:
            float_num = None
            print('error: unknown sign')

    return float_num


def file_accumulate(buffer_data,old_data,read_count,read_columns):
    f_read=open(buffer_data,"r")
    buffer_list=[]
    for line in f_read:
        if "#" in line:
            continue
        buffer_list_line=[]
        tick=[]
        tick=line.split(" ")
        for i in range(len(tick)):
            if len(tick[i])>0:
                buffer_list_line.append(tick[i])
        buffer_list.append(buffer_list_line)
    if len(buffer_list)>2:
#       print("%s has read!"%buffer_data) 
       read_count+=1
    if len(buffer_list)<=2:
       empty_file.append(buffer_data)
       print("%s is an empty file!"%buffer_data)
       return [old_data,read_count]
    if len(old_data)==0:
        for i in range(len(buffer_list)):
            old_data_line=[]
            for j in range(len(buffer_list[0])):
                old_data_line.append(0)
            old_data.append(old_data_line)
    elif len(old_data)!=0:
        for i in range(len(buffer_list)):
            for j in range(read_columns):
                if 'E' not in old_data[i][j]:
                    if 'e' not in old_data[i][j]:
                        buffer_list[i][j]=old_data[i][j]
                    elif 'e' in old_data[i][j]:
                        buffer_list[i][j]='%.3e'%(Etofloat(buffer_list[i][j])+Etofloat(old_data[i][j]))
                elif 'E' in old_data[i][j]:
                    buffer_list[i][j]='%.3e'%(Etofloat(buffer_list[i][j])+Etofloat(old_data[i][j]))
#print(buffer_list)
    f_read.close()
    return [buffer_list,read_count]

def out_file(file_list,output_file,read_columns):
    #print("*****************\n")
    #print(file_list) 
    old_data=[]
    empty_file=[]
    read_count=0
    for i in range(len(file_list)):
        old_data,read_count=file_accumulate(file_list[i],old_data,read_count,read_columns)
    #    print(file_list[i])
    output_data=[]
    for i in range(len(old_data)):
        output_data_line=[]
        for j in range(len(old_data[0])):
            if j==0:
                output_data_line.append(str(float(old_data[i][j])-Fermi_E))
            else:
                output_data_line.append(old_data[i][j])
        output_data.append(output_data_line)
    
    
    if os.path.exists(output_file)==1:
        os.system("rm %s"%output_file)
    os.mknod(output_file)
    f_write=open(output_file,"w+")
    #buffer_array=np.array(buffer_list)
    #buffer_array_avail=buffer_array[:,0]
    buffer_array_avail=[]
    #print("it is length %d"%len(buffer_list[0]))
    for i in range(len(output_data)):
        buffer_array_avail_line=[]
        for j in range(read_columns):
            buffer_array_avail_line.append("%.7f"%float(output_data[i][j])+"  ")
            if j ==read_columns-1:   
                buffer_array_avail_line.append('\n')
        buffer_array_avail.append(buffer_array_avail_line)
    #print(buffer_array_avail)
    for i in range(len(buffer_array_avail)):
        f_write.writelines(buffer_array_avail[i])
    f_write.close()
    if len(empty_file)>0:
        print("WARNING: Those files are empty or somehow crashed so that they were not read.Please check again!")
        print(empty_file)
    print("congratulation,there is %d files accumulated!"%read_count)
    print(f"{output_file} generated!")    

#detected_dic,atomic_dic_files,draw_index,read_lines_from_pdos_lst is available in below section
error=0
if os.path.isfile("out_scf_%s"%str_name)==1:
    Fermi_E=float(os.popen("grep Fermi out_scf_%s"%str_name).readlines()[-1].split("is")[-1].split("ev")[0].strip())
elif os.path.isfile("out_relax_%s"%str_name)==1:
    Fermi_E=float(os.popen("grep Fermi out_relax_%s"%str_name).readlines()[-1].split("is")[-1].split("ev")[0].strip())
else:
    error=1
    print("No relaxtion file or scf file found!")

if error==0:
    dup_index=find_duplicates([x[1] for x in result_his_lst])[::-1]
    #print([x[1] for x in result_his_lst])
    #print(dup_index)
    for i,v in enumerate(result_his_lst):
        if i in [x[1] for x in dup_index]:
            print("removing:",i,v)
            result_his_lst.remove(v)
    found_file_index=[]
    for i,v in enumerate(result_his_lst):
        for j,m in enumerate(detected_dic):
            if m==v[1]:
                print("###read!",m,v[0],draw_index[j],input_seg[j])
                found_file_index.append(j)

    count_output=0
    if len(result_his_lst)!=0: 
        original_files=max([int(x[0].split(output_file_head)[-1]) for x in result_his_lst])
    else:
        original_files=0
    for j in range(len(atomic_dic_files)):
        if j not in found_file_index:
            print("###Calculating...",detected_dic[j],output_file_head+str(original_files+1+count_output),draw_index[j],input_seg[j])
            out_file(atomic_dic_files[j],output_file_head+str(original_files+1+count_output),read_lines_from_pdos_lst[j])
            result_his_lst.append([output_file_head+str(original_files+1+count_output),detected_dic[j]])
            count_output+=1

    f_output_data=open(result_file,"w+")
    for i in result_his_lst:
        f_output_data.writelines(f"{i[0]}\t{i[1]}\n")
    f_output_data.close()
