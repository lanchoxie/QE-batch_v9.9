import os
import numpy as np
import sys
import re
import time

split_sybol_virtual="-"

vaspfile=sys.argv[1]


molecule = vaspfile.split(".vasp")[0]            

input2 = ".vasp"


dir_file=(os.popen("pwd").read())
dir=max(dir_file.split('\n'))


def read_vasp_single_file( filename ):
    with open(filename,'r') as f:
        raw = f.readlines()
    coorf = []; name = []; fix = []; lat = np.zeros( (3,3) )
    scale_factor = float(raw[1].split()[0])
    for order,i in enumerate(raw[2:5]):
        tmp_ = [float(j.strip().strip('\n')) for j in i.split()]
        lat[order,:] = np.array( tmp_ )
    name_set = raw[5].split()
#    for names_i in range(len(name_set)):
#        name_set[names_i]=name_set[names_i].replace(".","")
    name_count = [int(j) for j in raw[6].split()]
    name_count_spec = len(name_set)
    natom = sum( name_count )
    for iname in range(name_count_spec):
        for j in range(name_count[iname]):
            name.append( name_set[iname])
    for i in raw[8:natom+8]:
        tmp_ = [j.strip().strip('\n') for j in i.split()]
        tmp_2 = list(map(float, tmp_[0:3]))
        coorf.append( tmp_2 )
        coord = np.array(coorf)
    return np.array(name), lat, np.array(coord),natom,name_set

def write_func(name2,lat2,coord2,natom2):
#    print the way input file aquires
    writethings=[]
    writethings.append("CELL_PARAMETERS {alat}\n")
    for i_cell_par in range(0,3):
        str="   "
        for j_cell_par in range(0,3):
            str+="%.10f"%lat2[i_cell_par][j_cell_par]
            str+="      "
        writethings.append(str+'\n')
        str=""
    writethings.append("ATOMIC_POSITIONS (crystal)\n")
    for i in range(natom2):
        if split_sybol_virtual in name2[i]:
            mixed_ele_spe=[]
            element_i_s=[x for x in name2[i].split(split_sybol_virtual) if len(x)>0]
            for jj in element_i_s:
                if (element_i_s.index(jj) % 2) == 0:
                    mixed_ele_spe.append(jj)
            str1="" 
            str1="%s      %.9f     %.9f      %.9f "%((mixed_ele_spe[0]+mixed_ele_spe[1]),coord2[i][0],coord2[i][1],coord2[i][2])
        else:
            str1="" 
            #str1="%s      %.9f     %.9f      %.9f "%(name2[i].replace(".",split_sybol_virtual),coord2[i][0],coord2[i][1],coord2[i][2])
            str1="%s      %.9f     %.9f      %.9f "%(name2[i],coord2[i][0],coord2[i][1],coord2[i][2])
        writethings.append(str1+'\n')
#   print(writethings)
    return writethings

input=molecule+input2
name1,lat1,coord1,natom1,name_set2=read_vasp_single_file(input)
for i in write_func(name1,lat1,coord1,natom1):
	print(i.strip("\n"))


