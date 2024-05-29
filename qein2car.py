import sys
import os
import re
from collections import defaultdict
import numpy as np

if len(sys.argv)<2:
    raise ValueError("Plz specifize the input file you want to calculate!")

if len(sys.argv)<3:
    raise ValueError("Plz specifize the output structure format! (vasp/xsf)")

input_file=sys.argv[1]
output_format=sys.argv[2]

if not os.path.isfile(input_file):
    raise ValueError(f"{input_file} not exist!")

if output_format not in ["xsf","vasp"]:
    raise ValueError(f"{sys.argv[2]} not in [xsf,vasp]!!")

dir_read="./"
input_cont=open(input_file).readlines()
input_data="".join(input_cont)
for i in input_cont:
    lattice_raw=os.popen(f"grep -A 3 \"CELL_PARAMETERS\" {input_file} | tail -n 3").readlines()
print("*"*20,lattice_raw,"*"*20)
patt_coord=r"\s*([A-Za-z]+)\s+([-.0-9]+)\s+([-.0-9]+)\s+([-.0-9]+)"
configurations = defaultdict(list)
for line in input_data.split("\n"):
    match = re.search(patt_coord, line)
    if match:
        #print(line)
        ele, posi1, posi2, posi3 = match.groups()
        configurations[ele].append([posi1,posi2,posi3])

str_name=input_file
if output_format == "vasp":
    head=[f"POSCAR CARTESIAN FILE OF {str_name}\n","1.0\n"]
    if not os.path.isfile(dir_read+str_name+".vasp"):
        f_new=open(dir_read+str_name+".vasp", 'w')
        f_new.close()
    f=open(dir_read+str_name+".vasp","w+")
    for i in head:
        f.writelines(i)
    for i in lattice_raw:
        f.writelines(i)
    f.writelines(" ".join([i for i in configurations])+"\n")
    f.writelines(" ".join([str(len(v)) for i,v in configurations.items()])+"\n")
    f.writelines("Direct\n")
    for i,v in configurations.items():
        for j in v:
            f.writelines("    ".join(j)+"\n")
    f.close()
    print(f"{str_name}.vasp created!!")

elif output_format == "xsf":
    lattice_vectors = np.array([list(map(float, vec.split())) for vec in lattice_raw])
    #print(lattice_vectors)
    configurations_car={}
    for i,v in configurations.items():
        configurations_car[i]=[]
        for m in v:
            #print(m)
            configurations_car[i].append(np.dot(np.array([float(x) for x in m]),lattice_vectors))

    head=[f"DIM-GROUP\n","     3     1\n","PRIMVEC\n"]
    if not os.path.isfile(dir_read+str_name+".xsf"):
        f_new=open(dir_read+str_name+".xsf", 'w')
        f_new.close()
    f=open(dir_read+str_name+".xsf","w+")
    for i in head:
        f.writelines(i)
    for i in lattice_raw:
        f.writelines(i)
    f.writelines("PRIMCOORD\n")
    f.writelines(f"{    sum([(len(v)) for i,v in configurations.items()])}    1"+"\n")
    for i,v in configurations_car.items():
        for j in v:
            f.writelines(i+"  "+"    ".join([str(x) for x in j])+"\n")
    f.close()
    print(f"{str_name}.xsf created!!")
