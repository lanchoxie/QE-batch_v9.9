# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 23:39:53 2023

@author: xiety
"""

from __future__ import annotations
from pymatgen.core.periodic_table import Element
# Import modules
from pymatgen.io.xcrysden import XSF

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core import Structure
from collections import defaultdict
import pandas as pd
import numpy as np
import subprocess
import sys
import pickle
import os

#The center atom
#center_atom_number = 4  #Not list index, the atom number in vesta!

#selected_element_lst=['Ni','Li','Co','Mn','Ti','O']
#surrounding_type_lst=['Li','TM','Li','Li','Li','tot'] #'TM' for transition metal,'Li','O',tot for no element specified
surround_specified_lst=[[['Li','Li'],['O','tot'],['TM','Li']], 
                    [['Li','TM'],['O','tot'],['TM','TM']],
                    [['Li','O'],['O','tot'],['TM','O']]]#specify the surrounding element,['center_element','surrounding_element']
#WARNING:when doping other element except for Transition Metal, I suggest change the code, else  there will be some error
#The neighbor element type
O_neighbor=Composition("O")
Li_neighbor=Composition("Li")
TM_ele=['Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn']
TM_neighbor=[Composition(x) for x in TM_ele]


str_in=sys.argv[1]
df_dir='data.save'
str_dir=f'{str_in}'

df_all_file = f'{df_dir}/bader_charge_of_{str_in}.data'
str_file = f"{str_dir}/out_relax_{str_in}.xsf"
file_old = f"out_relax_{str_in}.xsf" #used in generating files
script_dir=sys.path[0]+"/"

infile=str_file

    
print("*************STUCTURE**************")
print(str_in)


#structure = Poscar.from_file(infile).structure

class XSF_1:
    """
    Class for parsing XCrysden files.
    """

    def __init__(self, structure):
        """
        :param structure: Structure object.
        """
        self.structure = structure

    def to_string(self, atom_symbol=True):
        """
        Returns a string with the structure in XSF format
        See http://www.xcrysden.org/doc/XSF.html

        Args:
            atom_symbol (bool): Uses atom symbol instead of atomic number. Defaults to True.
        """
        lines = []
        app = lines.append

        app("CRYSTAL")
        app("# Primitive lattice vectors in Angstrom")
        app("PRIMVEC")
        cell = self.structure.lattice.matrix
        for i in range(3):
            app(f" {cell[i][0]:.14f} {cell[i][1]:.14f} {cell[i][2]:.14f}")

        cart_coords = self.structure.cart_coords
        app("# Cartesian coordinates in Angstrom.")
        app("PRIMCOORD")
        app(f" {len(cart_coords)} 1")

        for site, coord in zip(self.structure, cart_coords):
            sp = site.specie.symbol if atom_symbol else f"{site.specie.Z}"
            x, y, z = coord
            app(f"{sp} {x:20.14f} {y:20.14f} {z:20.14f}")

        return "\n".join(lines)

    @classmethod
    def from_string(cls, input_string, cls_=None):
        """
        Initialize a `Structure` object from a string with data in XSF format.

        Args:
            input_string: String with the structure in XSF format.
                See http://www.xcrysden.org/doc/XSF.html
            cls_: Structure class to be created. default: pymatgen structure
        """
        # CRYSTAL                                        see (1)
        # these are primitive lattice vectors (in Angstroms)
        # PRIMVEC
        #    0.0000000    2.7100000    2.7100000         see (2)
        #    2.7100000    0.0000000    2.7100000
        #    2.7100000    2.7100000    0.0000000

        # these are conventional lattice vectors (in Angstroms)
        # CONVVEC
        #    5.4200000    0.0000000    0.0000000         see (3)
        #    0.0000000    5.4200000    0.0000000
        #    0.0000000    0.0000000    5.4200000

        # these are atomic coordinates in a primitive unit cell  (in Angstroms)
        # PRIMCOORD
        # 2 1                                            see (4)
        # 16      0.0000000     0.0000000     0.0000000  see (5)
        # 30      1.3550000    -1.3550000    -1.3550000

        lattice, coords, species = [], [], []
        lines = input_string.splitlines()
        for i, line in enumerate(lines):
            if "PRIMVEC" in line:
                for j in range(i + 1, i + 4):
                    lattice.append([float(c) for c in lines[j].split()])

            if "PRIMCOORD" in line:
                num_sites = int(lines[i + 1].split()[0])

                for j in range(i + 2, i + 2 + num_sites):
                    tokens = lines[j].split()
                    #Z = Element(tokens[0]).Z if tokens[0].isalpha() else int(tokens[0])
                    Z = Element(''.join([i for i in tokens[0] if not i.isdigit()])).Z
                    species.append(Z)
                    coords.append([float(j) for j in tokens[1:4]])
                break
        else:
            raise ValueError("Invalid XSF data")

        if cls_ is None:
            from pymatgen.core.structure import Structure

            cls_ = Structure

        s = cls_(lattice, species, coords, coords_are_cartesian=True)
        return XSF(s)


f1=open(infile).readlines()
f2=""
for i in f1:
    f2+=i
xsf = XSF_1.from_string(f2)
structure = xsf.structure

ele_spe=list(set([str(structure.sites[i].species)[:-1] for i in range(len(structure))]))
print("*************ELEMENT**************")
print(ele_spe)

if os.path.isfile(f"{df_dir}/{str_in}_cluster_data.pkl"):
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    loaded_data={}
    with open(f"{df_dir}/{str_in}_cluster_data.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    spe_df=loaded_data['spe_df']
else:
    ele_metal=ele_spe.copy()
    #ele_metal.remove("O")
    save_data={}
    spe_df={}
    for ele_i in ele_metal:
        if ele_i != 'O':
            jj=os.popen(f"python {script_dir}clustering-xsf.py {str_in} {ele_i} 0 1").readlines()
            print(f"python {script_dir}clustering-xsf.py {str_in} {ele_i} 0 1")
        elif ele_i=="O":
            jj=os.popen(f"python {script_dir}clustering-xsf.py {str_in} {ele_i} 0 0").readlines()
            print(f"python {script_dir}clustering-xsf.py {str_in} {ele_i} 0 0")
        count=0
        for i in jj:
            if "###" in i:
                spe_df[f'{ele_i}#{count+1}']=[int(numbers.strip()) for numbers in i.strip("\n").split('[')[-1].split(']')[0].split(',')]
                count+=1
    save_data['spe_df']=spe_df
    # 保存数据到文件
    with open(f"{df_dir}/{str_in}_cluster_data.pkl", "wb") as file:
        pickle.dump(save_data, file)

print("*************CLUSTERS**************")
element_cluster_number={}
for key,value in spe_df.items():
    print(key,value)
    if key.split("#")[0] not in [key_in for key_in, values_in in element_cluster_number.items()]:
        element_cluster_number[key.split("#")[0]]=1
    elif key.split("#")[0] in [key_in for key_in, values_in in element_cluster_number.items()]:
        element_cluster_number[key.split("#")[0]]+=1
print("*************CLUSTERS_STATISTIC**************")
print(element_cluster_number)
#print(spe_df)

def find_cluster(index):
    matching_keys = [key for key, values in spe_df.items() if index in values]
    if matching_keys:
        return matching_keys[0]
    else:
        raise ValueError(f'{index} not found!')

#print(find_cluster(100))

vec_a=structure.lattice.matrix[0]
vec_b=structure.lattice.matrix[1]
vec_c=structure.lattice.matrix[2]
A_vec = np.vstack([vec_a, vec_b, vec_c]).T
A_inv = np.linalg.inv(A_vec)
pbc_cell_333=[-1,0,1]

def neighbor_get(center_atom_number,surrounding_type):

    center_atom_index = center_atom_number - 1    
    ele_info_neighbor=[]
    for i in range(len(structure)):
        dis_i=[]
        for a in pbc_cell_333: #contains the 3x3x3 supercell atoms and only get the minimal distance
            for b in pbc_cell_333:
                for c in pbc_cell_333:
                    site_i=np.array(structure.sites[i].coords)+vec_a*a+vec_b*b+vec_c*c
                    #calculate the distance between the neighbors and the center to select the nearest neighbors
                    dis_i.append(np.linalg.norm(site_i - np.array(structure.sites[center_atom_index].coords)))
        dis=min(dis_i)
        if dis!=0:#exclude self
            ele_info_neighbor.append([structure.sites[i].species,(i+1),structure.sites[i].coords,dis])
    
    ele_info_neighbor=sorted(ele_info_neighbor,key=lambda x:x[-1])#list sorted by distance
    neighbor_O_tot=[]
    neighbor_Li_tot=[]
    neighbor_TM_tot=[]
    
    for i in ele_info_neighbor:
        if i[0]==O_neighbor:
            neighbor_O_tot.append(i)
        elif i[0]==Li_neighbor:
            neighbor_Li_tot.append(i)    
        elif i[0] in TM_neighbor:
            neighbor_TM_tot.append(i)
    
    neighbor_O_sel=neighbor_O_tot[:6]
    neighbor_Li_sel=neighbor_Li_tot[:6]
    neighbor_TM_sel=neighbor_TM_tot[:6]
    neighbor_tot_sel=ele_info_neighbor[:6]
    #print("****************Selected Atom********************")
    #print(structure.sites[center_atom_index].species,(center_atom_index+1),find_cluster(center_atom_index+1),structure.sites[center_atom_index].coords)
    #print("*************************************************")
    atom_environment={}
    surr_atom=[]
    if surrounding_type=="TM":
        for i in neighbor_TM_sel:
            surr_atom.append(f'{i[1]}-{find_cluster(i[1])}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="Li":
        for i in neighbor_Li_sel:
            surr_atom.append(f'{i[1]}-{find_cluster(i[1])}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="O":
        for i in neighbor_O_sel:
            surr_atom.append(f'{i[1]}-{find_cluster(i[1])}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    elif surrounding_type=="tot":
        for i in neighbor_tot_sel:
            surr_atom.append(f'{i[1]}-{find_cluster(i[1])}')
            #print(i[0],i[1],find_cluster(i[1]),i[2],i[3])
    return surr_atom

def get_datas(surround_specified):
    
    #read surrounding element from surround_specified and was used in neighbor_get(center_atom_number,surrounding_type)
    selected_element_lst=ele_spe.copy()
    surrounding_type_lst=[]
    for i in selected_element_lst:
        if i in [x[0] for x in surround_specified]:
            surrounding_type_lst.append(surround_specified[[x[0] for x in surround_specified].index(i)][1])
        else:
            surrounding_type_lst.append(surround_specified[[x[0] for x in surround_specified].index('TM')][1])
    
    nearest_environments_lst=[]
    for i,selected_element in enumerate(selected_element_lst):
        surrounding_type=surrounding_type_lst[i]
        selected_atom_number=[]
        for i in range(len(structure)):
            if structure.sites[i].species==Composition(selected_element):
                selected_atom_number.append(i+1) # store the selected atomic number which is the element you settled
        #print(selected_atom_number)
        if selected_atom_number==[]:
            raise ValueError(f"{selected_element} not found in {ele_spe}!")  
        print(f"#*************SURROUND_ATOM:{selected_element}-{surrounding_type}**************")
        for i in selected_atom_number:
            #print(f'{i}-{find_cluster(i)}:',f'{neighbor_get(i,surrounding_type)}')
            nearest_environments_lst.append(f'{i}-{find_cluster(i)}\t%s'%(str(neighbor_get(i,surrounding_type)).replace(', ','; ')))
    nearest_environments_lst=sorted(nearest_environments_lst,key=lambda x:int(x.split("-")[0]))
    return nearest_environments_lst

output_datas=[]
out_data_0=get_datas(surround_specified_lst[0])
out_data_1=get_datas(surround_specified_lst[1])
out_data_2=get_datas(surround_specified_lst[2])
for i,v in enumerate(out_data_0):
    seg1=out_data_1[i].split("\t")[1]
    seg2=out_data_2[i].split("\t")[1]
    output_datas.append(v+f"\t{seg1}\t{seg2}")
f_out=open(f"{df_dir}/SURROUNDING_ATOMS_of_{str_in}.txt","w+")
f_out.writelines("#Center_Atom_env\tSurround_Li_env\tSurround_TM_env\tSurround_O_env\n")
for i in output_datas:
    print(i)
    f_out.writelines(i+"\n")
print(f"{df_dir}/SURROUNDING_ATOMS_of_{str_in}.txt generated!")
