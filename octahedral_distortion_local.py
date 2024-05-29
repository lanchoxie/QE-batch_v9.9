import os
import sys
import numpy as np
from pymatgen.core.periodic_table import Element
# Import modules
from pymatgen.io.xcrysden import XSF
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core import Structure
from collections import Counter


atom_1=sys.argv[1]
atom_2=sys.argv[2]
str_name=sys.argv[3]

str_file = f"{str_name}/out_relax_{str_name}.xsf"
script_dir=sys.path[0]+"/"

# 初始化Ni，Mn，Ti，Co的计数为0
counts = {'Ni': 0, 'Co': 0, 'Mn': 0, 'Ti': 0, 'Zr':0, 'Mg':0, 'Al':0}


file_i=f"data.save/Separate_Surrouding_List_of_{str_name}.txt"
f=open(file_i).readlines()

line_data=[]
for i in f:
    line_data.append(i.split("\n")[0].split("\t"))


file_tm=f"data.save/SURROUNDING_ATOMS_of_{str_name}.txt"
f_tm=open(file_tm).readlines()

line_data_tm=[]
for i in f_tm:
    line_data_tm.append(i.split("\n")[0].split("\t"))
#print(line_data[0])
infile=str_file

    
#print("*************STUCTURE**************")
#print(str_name)
#print("***********************************")


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


O_env={}
for i in line_data:
    O_env_i=[]
    index_j=line_data[0].index("最近邻O环境类型")
    mm=i[index_j].strip("[").strip("]").split("+ ")
    nn=[i.strip("'") for i in mm]
    O_env[i[0]]=[]
    O_env_i.extend(nn)
    O_env[i[0]]=O_env_i

TM_env={}
for i in line_data_tm:
    TM_env_i=[]
    index_j=line_data_tm[0].index("Surround_TM_env")
    mm=i[index_j].strip("[").strip("]").split("; ")
    nn=[i.strip("'") for i in mm]
    TM_env[i[0]]=[]
    TM_env_i.extend(nn)
    TM_env[i[0]]=TM_env_i

#print(TM_env)

#for i in O_env:
#    print(i)
O_surr_1=[]
O_surr_2=[]
for key,v in O_env.items():
    if key.split("#")[0]==atom_1:
        O_surr_1=v
    if key.split("#")[0]==atom_2:
        O_surr_2=v

print(atom_1,atom_2)
print(O_surr_1,O_surr_2)

cross_O=[]
for i in O_surr_1:
    if i in O_surr_2:
        cross_O.append(i)
print(cross_O)


TM_surr_1=[]
TM_surr_2=[]
for key,v in TM_env.items():
    if key.split("#")[0]==atom_1:
        TM_surr_1=v
    if key.split("#")[0]==atom_2:
        TM_surr_2=v
#print(TM_surr_1,TM_surr_2)

cross_TM=[]
for i in TM_surr_1:
    if i in TM_surr_2:
        cross_TM.append(i)
#print(TM_surr_1)
#print(TM_surr_2)
#print(cross_TM)


vec_a=structure.lattice.matrix[0]
vec_b=structure.lattice.matrix[1]
vec_c=structure.lattice.matrix[2]
A_vec = np.vstack([vec_a, vec_b, vec_c]).T
A_inv = np.linalg.inv(A_vec)
pbc_cell_333=[-1,0,1]

def dis_get(center_atom_number,O_atom_number):

    center_atom_index = center_atom_number - 1    
    ele_info_neighbor=[]
    for i in O_atom_number:
        dis_i=[]
        for a in pbc_cell_333: #contains the 3x3x3 supercell atoms and only get the minimal distance
            for b in pbc_cell_333:
                for c in pbc_cell_333:
                    site_i=np.array(structure.sites[i].coords)+vec_a*a+vec_b*b+vec_c*c
                    #calculate the distance between the neighbors and the center to select the nearest neighbors
                    dis_i.append(np.linalg.norm(site_i - np.array(structure.sites[center_atom_index].coords)))
        dis=min(dis_i)
        if dis!=0:#exclude self
            #ele_info_neighbor.append([structure.sites[i].species,(i+1),structure.sites[i].coords,dis])
            ele_info_neighbor.append([str(i+1)+"-"+str(structure.sites[i].species)[:-1],f"{dis:.6f}"])
    
    return ele_info_neighbor

O_cross_number=[(int(x.split("-")[0])-1) for x in cross_O]
input_atom={atom_1.split("#")[0]:dis_get(int(atom_1.split("-")[0]),O_cross_number),atom_2.split("#")[0]:dis_get(int(atom_2.split("-")[0]),O_cross_number)}
output_atom={"O_X_type":"--".join([i.split("#")[0] for i in cross_O])}
keys=[key for key,v in input_atom.items()]
key_ele=[i.split("-")[1] for i in keys]
#print(O_cross_number)
O_O_bonds=dis_get((O_cross_number[0]+1),[O_cross_number[1]])[0]
other_O_info={"O-O_bonds_X":O_O_bonds[1],f"mean-{key_ele[0]}-XO":f"{np.mean([float(i[1]) for i in input_atom[keys[0]]]):.6f}",f"mean-{key_ele[1]}-XO":f"{np.mean([float(i[1]) for i in input_atom[keys[1]]]):.6f}",f"max-{key_ele[0]}-XO":f"{np.max([float(i[1]) for i in input_atom[keys[0]]])}",f"max-{key_ele[1]}-XO":f"{np.max([float(i[1]) for i in input_atom[keys[1]]])}",f"min-{key_ele[0]}-XO":f"{np.min([float(i[1]) for i in input_atom[keys[0]]])}",f"min-{key_ele[1]}-XO":f"{np.min([float(i[1]) for i in input_atom[keys[1]]])}"}

#print(other_O_info)

TM_cross_number=[(int(x.split("-")[0])-1) for x in cross_TM]
TM_cross_ele=[(x.split("#")[0].split("-")[1]) for x in cross_TM]
input_atom_tm={atom_1.split("#")[0]:dis_get(int(atom_1.split("-")[0]),TM_cross_number),atom_2.split("#")[0]:dis_get(int(atom_2.split("-")[0]),TM_cross_number)}
print(input_atom_tm)
key_tm=[key for key,v in input_atom_tm.items()]
key_tm_ele=[i.split("-")[1] for i in key_tm]

# 使用Counter来统计列表中每个元素的出现次数
element_counter = Counter(TM_cross_ele)
# 更新counts字典中的计数
counts.update(element_counter)
TM_infos_0={f"{element}_on_cross": str(count) for element, count in counts.items()}
#TM_infos_temp={f"{key_ele[0]}-X-{TM_cross_ele[0]}-dis":f"{input_atom_tm[keys[0]][0][1]}",f"{key_ele[0]}-X-{TM_cross_ele[1]}-dis":f"{input_atom_tm[keys[0]][1][1]}",f"{key_ele[1]}-X-{TM_cross_ele[0]}-dis":f"{input_atom_tm[keys[1]][0][1]}",f"{key_ele[1]}-X-{TM_cross_ele[1]}-dis":f"{input_atom_tm[keys[1]][1][1]}"}
TM_infos_1={"TM_X_type":"--".join([i.split("#")[0] for i in cross_TM]),f"{key_ele[0]}-X-TM1-dis":f"{input_atom_tm[keys[0]][0][1]}",f"{key_ele[0]}-X-TM2-dis":f"{input_atom_tm[keys[0]][1][1]}",f"{key_ele[1]}-X-TM1-dis":f"{input_atom_tm[keys[1]][0][1]}",f"{key_ele[1]}-X-TM2-dis":f"{input_atom_tm[keys[1]][1][1]}"}

TM_TM_bonds=dis_get((TM_cross_number[0]+1),[TM_cross_number[1]])[0]
other_TM_info={"TM-TM_bonds_X":TM_TM_bonds[1],f"mean-{key_ele[0]}-XTM":f"{np.mean([float(i[1]) for i in input_atom_tm[keys[0]]]):.6f}",f"mean-{key_ele[1]}-XTM":f"{np.mean([float(i[1]) for i in input_atom_tm[keys[1]]]):.6f}",f"max-{key_ele[0]}-XTM":f"{np.max([float(i[1]) for i in input_atom_tm[keys[0]]])}",f"max-{key_ele[1]}-XTM":f"{np.max([float(i[1]) for i in input_atom_tm[keys[1]]])}",f"min-{key_ele[0]}-XTM":f"{np.min([float(i[1]) for i in input_atom_tm[keys[0]]])}",f"min-{key_ele[1]}-XTM":f"{np.min([float(i[1]) for i in input_atom_tm[keys[1]]])}"}
#print(TM_infos_1)

features="###"+"\t".join([key for key,v in TM_infos_0.items()])+"\t"+"\t".join([key for key,v in TM_infos_1.items()])+"\t"+"\t".join([key for key,v in other_TM_info.items()])+"\t"+"\t".join([key for key,v in output_atom.items()])+"\t"+"\t".join([key for key,v in other_O_info.items()])
values='###'

#print("###"+"\t".join([key for key,v in TM_infos_0.items()])+"\t"+"\t".join([key for key,v in TM_infos_1.items()])+"\t"+"\t".join([key for key,v in other_TM_info.items()]))
#print("###"+"\t".join([v for key,v in TM_infos_0.items()])+"\t"+"\t".join([v for key,v in TM_infos_1.items()])+"\t"+"\t".join([v for key,v in other_TM_info.items()]))


if len(cross_TM)==0:
    #print("###"+"\t".join([key for key,v in output_atom.items()])+"\t"+"\t".join([key for key,v in other_O_info.items()]))
    #print("###"+"No cross O\t"*(len(output_atom)+len(other_O_info)))
    values+="\t"+"No_cross_TM\t"*(len(TM_infos_0)+len(TM_infos_1)+len(other_TM_info))
else:
    #print("###"+"\t".join([key for key,v in output_atom.items()])+"\t"+"\t".join([key for key,v in other_O_info.items()]))
    #print("###"+"\t".join([v for key,v in output_atom.items()])+"\t"+"\t".join([v for key,v in other_O_info.items()]))
    values+="\t"+"\t".join([v for key,v in TM_infos_0.items()])+"\t"+"\t".join([v for key,v in TM_infos_1.items()])+"\t"+"\t".join([v for key,v in other_TM_info.items()])

if len(cross_O)==0:
    #print("###"+"\t".join([key for key,v in output_atom.items()])+"\t"+"\t".join([key for key,v in other_O_info.items()]))
    #print("###"+"No cross O\t"*(len(output_atom)+len(other_O_info)))
    values+="\t"+"No_cross_O\t"*(len(output_atom)+len(other_O_info))
else:
    #print("###"+"\t".join([key for key,v in output_atom.items()])+"\t"+"\t".join([key for key,v in other_O_info.items()]))
    #print("###"+"\t".join([v for key,v in output_atom.items()])+"\t"+"\t".join([v for key,v in other_O_info.items()]))
    values+="\t"+"\t".join([v for key,v in output_atom.items()])+"\t"+"\t".join([v for key,v in other_O_info.items()])

print(features)
print(values)
#if len(input_atom_tm)==0:
#    print("###"+"No cross O")
#else:
#    print("###"+)
