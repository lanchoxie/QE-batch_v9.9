import numpy as np
import os
import math
import sys

#for p orbital,it is  pz px py
#for d orbital,it is  dz2 dzx dzy dx2-y2 dxy


common_str_in_output_file="BTO.pdos_atm#"
input_str=sys.argv[1]
if "-" in input_str:
    common_str_in_output_file2=str(sys.argv[1]).split("-")[0]
    orbit_num=str(sys.argv[1]).split("-")[1]
    orbit_typ=str(sys.argv[1]).split("-")[2]
else:
    common_str_in_output_file2=str(sys.argv[1]).strip()
    orbit_num='not found'
    orbit_typ='not found'

print(common_str_in_output_file2)
output_file_tot="accumulated_pdos_file"+common_str_in_output_file2

dir_file=(os.popen("pwd").read())
dir_current=max(dir_file.split('\n'))

file_list_tot=[]
file_input_tag=[]
for root,dirs,files in os.walk(dir_current):
    for file in files:
        if "in_scf" in file or "in_relax" in file:
            file_input_tag.append(file)
        if common_str_in_output_file in file:
            if common_str_in_output_file2 in file.split('#')[1]:
                file_list_tot.append(file)
if len(file_input_tag)==0:
    raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")

spin_mode=0
input_tag=open(file_input_tag[0]).readlines()
for lines in input_tag:
    if "nspin" in lines and "2" in lines and "!" not in lines:
        spin_mode=1   


file_dic=[]
for i,files in enumerate(file_list_tot):
    file_tail_tick=files.split("#")[-1].strip("\n").strip(")").split("(")
    file_tail=file_tail_tick[0]
    if file_tail not in [x[0] for x in file_dic]:
        file_dic_list=[files]
        file_dic.append([file_tail,file_tail_tick[1],file_dic_list])
    elif file_tail in [x[0] for x in file_dic]:
        file_dic[[x[0] for x in file_dic].index(file_tail)][-1].append(files)

def sort_orbit_num(orbit):
    return ''.join([i for i in orbit[0] if i.isdigit()])
def sort_atom_num(atom):
    return ''.join([i for i in atom.split("#")[1] if i.isdigit()])

for i in file_dic:
    i[2].sort(key=sort_atom_num)
file_dic.sort(key=sort_orbit_num)


if (orbit_num!='not found')&(orbit_typ!='not found'):
    orbit_found=0
    for i,v in enumerate(file_dic):
        if (orbit_num==v[0])&(orbit_typ==v[1]):
            orbit_found=1
    if orbit_found==0:
        raise ValueError(f"{common_str_in_output_file2}-{orbit_num}-{orbit_typ} not found!","These are the Orbit found:",[f"{common_str_in_output_file2}-{v[0]}-{v[1]}" for v in file_dic])

#for i,classes in enumerate(file_dic):
#    print(i,classes[0],classes[1],classes[2])

if len(file_list_tot)==0:
    print("the string input is not matched with any file!")

else:
#    print(file_list_tot)
    nothing=1


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
#    buffer_list.append('\n')
#print(buffer_list)
    f_read.close()
    return [buffer_list,read_count]

def out_file(file_list,output_file,read_columns):
    
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

error=0
if os.path.isfile("out_scf_%s"%str_name)==1:
    Fermi_E=float(os.popen("grep Fermi out_scf_%s"%str_name).readlines()[-1].split("is")[-1].split("ev")[0].strip())
elif os.path.isfile("out_relax_%s"%str_name)==1:
    Fermi_E=float(os.popen("grep Fermi out_relax_%s"%str_name).readlines()[-1].split("is")[-1].split("ev")[0].strip())
else:
    error=1
    print("No relaxtion file or scf file found!")
if error==0:
    if (orbit_num!='not found')&(orbit_typ!='not found'):
        for i,v in enumerate(file_dic):
            if (orbit_num==v[0])&(orbit_typ==v[1]):
                if orbit_typ=="s":
                    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",3+2*spin_mode)
                if orbit_typ=="p":
                    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",5+4*spin_mode)
                if orbit_typ=="d":
                    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",7+6*spin_mode)
    else:
        out_file(file_list_tot,output_file_tot,3)
    #for i,v in enumerate(file_dic):
        #if v[1]=="s":
        #    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",5)
        #if v[1]=="p":
        #    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",9)
        #if v[1]=="d":
        #    out_file(v[2],output_file_tot+f"-{v[0]}-{v[1]}",13)
