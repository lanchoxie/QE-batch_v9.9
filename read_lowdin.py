import re
import os
import sys
dir_read=sys.argv[1]
mode=sys.argv[2]

file_in=f"in_{mode}_{dir_read}"
file_out=f"out_{mode}_{dir_read}"

atomic_number=0

f_in_buffer=open(f"{dir_read}/{file_in}").readlines()
f_out_buffer=open(f"{dir_read}/{file_out}").readlines()

dir_file=(os.popen("pwd").read())
Current_dir=max(dir_file.split('\n'))
str_name=dir_read

read_start=0
read_count=1
atomic_species=[]
for i,v in enumerate(f_in_buffer):
    if "nat" in v:
        atomic_number=int(v.split("=")[-1].strip("\n").strip(",").strip())
    if 'ATOMIC_POSITIONS' in v:
        read_start=1
        continue
    if (read_start==1)&(read_count<atomic_number+1):
        read_count+=1
        atomic_species.append([x for x in v.split() if len(x)>0][0]) 
        
if atomic_number==0:
    raise ValueError(f"No nat found in input file of {dir_read}/{file_in}!!!")

data = open(f"{dir_read}/out_pdos_{dir_read}").readlines()

#Judge spin mode
file_input_tag=[]
for root,dirs,files in os.walk(Current_dir+"/"+str_name):
    for file in files:
        if "in_scf" in file or "in_relax" in file:
            file_input_tag.append(file)
if len(file_input_tag)==0:
    raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
spin_mode=0
input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
for lines in input_tag:
    if "nspin" in lines and "2" in lines and "!" not in lines:
        spin_mode=1

print(spin_mode)
# 正则表达式模式
pattern = r'\s+Atom #\s+(\d+):\s+total charge =\s+([\d.]+), s =\s+([\d.]+), p =\s+([\d.]+), d =\s+([\d.]+),'

pattern_s = r'\s+Atom #\s+(\d+):\s+total charge =\s+([\d.]+), s =\s+([\d.]+),.*'
pattern_p = r'\s+Atom #\s+(\d+):\s+total charge =\s+([\d.]+), p =\s+([\d.]+),.*'
pattern_d = r'\s+Atom #\s+(\d+):\s+total charge =\s+([\d.]+), d =\s+([\d.]+),.*'

if spin_mode==1:
    # 用于存储匹配结果的列表
    matches = []
    # 按行迭代数据并仅处理符合格式的行
    for line in data:
        if re.match(pattern, line.strip("\n")):
            # 如果行符合格式，将其添加到匹配列表中
            matches.extend(re.findall(pattern, line.strip("\n")))

    fnm_out=f"Lowdin_of_{dir_read}.txt"
    f_output=open(fnm_out,"w+")
    # 输出匹配结果
    f_output.writelines(f"#ATOM\tTotal_Charge\tS_Value\tP_Value\tD_Value\n")
    for match in matches:
        atom_number = match[0]
        total_charge = match[1]
        s_value = match[2]
        p_value = match[3]
        d_value = match[4]
        f_output.writelines(f"{atomic_species[int(atom_number)-1]}-{atom_number}\t{total_charge}\t{s_value}\t{p_value}\t{d_value}\n")
elif spin_mode==0:
    # 用于存储匹配结果的列表
    matches_s = []
    matches_p = []
    matches_d = []
    # 按行迭代数据并仅处理符合格式的行
    for line in data:
        if re.match(pattern_s, line.strip("\n")):
            # 如果行符合格式，将其添加到匹配列表中
            matches_s.extend(re.findall(pattern_s, line.strip("\n")))
        elif re.match(pattern_p, line.strip("\n")):
            # 如果行符合格式，将其添加到匹配列表中
            matches_p.extend(re.findall(pattern_p, line.strip("\n")))
        elif re.match(pattern_d, line.strip("\n")):
            # 如果行符合格式，将其添加到匹配列表中
            matches_d.extend(re.findall(pattern_d, line.strip("\n")))
    print(len(matches_s),len(matches_p),len(matches_d))
    fnm_out=f"Lowdin_of_{dir_read}.txt"
    f_output=open(fnm_out,"w+")
    # 输出匹配结果
    f_output.writelines(f"#ATOM\tTotal_Charge\tS_Value\tP_Value\tD_Value\n")
    for i,v in enumerate(matches_s):
        atom_number = matches_s[i][0]
        total_charge = matches_s[i][1]
        s_value = matches_s[i][2]
        if (matches_s[i][0]==matches_p[i][0])&(matches_s[i][1]==matches_p[i][1]):
            p_value = matches_p[i][2]
        else:
            p_value = "Nan"
        if (matches_s[i][0]==matches_d[i][0])&(matches_s[i][1]==matches_d[i][1]):
            d_value = matches_d[i][2]
        else:
            d_value = "Nan"
        f_output.writelines(f"{atomic_species[int(atom_number)-1]}-{atom_number}\t{total_charge}\t{s_value}\t{p_value}\t{d_value}\n")

print(f"{fnm_out} generated!")

