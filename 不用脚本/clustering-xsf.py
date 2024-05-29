# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 10:36:02 2023

@author: xiety
"""
from __future__ import annotations
from pymatgen.core.periodic_table import Element
from pymatgen.io.xcrysden import XSF
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pymatgen.core.composition import Composition
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.local_env import VoronoiNN
import pickle
import numpy as np
import subprocess
import sys
import os
#import time

#start_time=time.time()

str_in=sys.argv[1]
#clustering_element
clustering_element=sys.argv[2]
#The neighbor element type
O_neighbor=Composition("O")
Li_neighbor=Composition("Li")
ele_neighbor=O_neighbor

octahedral_clustering=1  #set this to 1 to introduce octahedral volume
i_th_pic=int(sys.argv[3])              #seek the clustering picture when set octahedral_clustering=1

#data_file_dir:
df_all = pd.read_csv(f'data.save/bader_charge_of_{str_in}.data', sep='\s+',header=None)

#str_file_dir:
#dir_="D:\\project\\谢琎老师锂电池\\第二次：Ti阻止了Li-Ni混排\\计算结果\\第二次：超交换与磁不开心-regular model\\SG15.PBE测试-Li-Ni互换\\vasp_files"
# read POSCAR 
poscar_file = f"{str_in}/out_relax_{str_in}.xsf"

verbose=0

#change the input element into pymatgen readable format
Clu_ele_str=Composition(clustering_element)
infile=poscar_file

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

str_read_in1=open(infile).readlines()
str_read_in2=""
for i in str_read_in1:
    str_read_in2+=i
xsf = XSF_1.from_string(str_read_in2)
structure = xsf.structure

clustering_ele_lst=[]
for i in structure:
    if i.species==Clu_ele_str:
        clustering_ele_lst.append(structure.index(i)+1)

def get_six_neighbor_atoms(neighbor_element_type,atomic_number):
    
    element_type=neighbor_element_type
    center_atom_index=atomic_number-1
    # create VoronoiNN object
    voronoi = VoronoiNN()
    # read the neighbor atom
    neighbors = voronoi.get_nn_info(structure, center_atom_index)

    if verbose==1:
        print("input data:")
        print("neighbor ele:",element_type,"atomic number",atomic_number)
    
        # check center atom COORD and ELEMENT
        #print([site['site'].coords for site in neighbors])
        #print([site['site'].species for site in neighbors])
        print("matched data:")
        #print(structure.sites[center_atom].coords,structure.sites[center_atom].species,structure.sites[center_atom].index)
        print(structure.sites[center_atom_index].coords,structure.sites[center_atom_index].species,center_atom_index+1)
        #print("*******")
        #print(neighbors)
    nb_coords=[site['site'].coords for site in neighbors]
    nb_spe=[site['site'].species for site in neighbors]
    nb_index=[site['site'].index for site in neighbors]
       
    nb_ele_coord=[]
    nb_ele_spe=[]
    nb_ele_index=[]
    
    #print(nb_spe[2],type(nb_spe[2]))
    for i in range(len(nb_spe)):
        #print(nb_spe[i])
        if nb_spe[i]==element_type:
            nb_ele_coord.append(nb_coords[i])
            nb_ele_spe.append(nb_spe[i])
            nb_ele_index.append(nb_index[i]+1)
           
    #print(nb_ele_spe)
    #print(nb_ele_coord)
    #print(nb_ele_index)    
    ele_info=[]
    for i in range(len(nb_ele_coord)):
        #calculate the distance between the neighbors and the center to select the nearest neighbors
        dis=np.linalg.norm(nb_ele_coord[i] - np.array(structure.sites[center_atom_index].coords))
        ele_info.append([nb_ele_spe[i],nb_ele_index[i],nb_ele_coord[i],dis])
    ele_info_min_six=sorted(ele_info, key=lambda x: x[-1])[:6]
    if verbose==1:
        print(ele_info_min_six)
        
    # calculate the volume of an octahedron    
    nearest_coord_lst=[x[2] for x in ele_info_min_six]
    #print(nb_site)
    # convert the coord of the neighbors into numpy array
    vertices = np.array(nearest_coord_lst)
    # To calculate the volume of an octahedron, you can use ConvexHull from the Scipy library
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    volume = hull.volume
    print(f"Calculated the volume of {atomic_number}: {volume}")
    return volume

#octahedral_vol_lst=[]
#for i in clustering_ele_lst:
    #octahedral_vol_lst.append(get_six_neighbor_atoms(O_neighbor,i))
def convert_to_ranges(lst_input):
    lst_data=[]
    for i in lst_input:
        lst_data.append(int("".join([j for j in i if j.isdigit()])))
    print(lst_data)
    lst=sorted(lst_data)
    
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


column_names = ['Element', 'Lowdin-ionic charge', 'Bader-ionic charge', 'Lowdin-valence charge', 'Bader-valence charge', 'Magnetic']
df_all.columns = column_names

element_row=df_all['Element'].tolist()
index_row=df_all['Magnetic'].tolist()
#print(element_row)
sele_ele_index=[]
sele_atom=[]
element_spe=[]
#print(len(element_row))

for i in element_row:
    #print(i)
    if i.split("-")[0] not in element_spe:
        #print(i)
        element_spe.append(i.split("-")[0])
        
print(element_spe)
if clustering_element in element_spe:
    pass
elif clustering_element not in element_spe:
    raise ValueError(f"{clustering_element} not in {element_spe}")

for i in element_row:
    if i.split("-")[0] == clustering_element:
        #print(i)
        if octahedral_clustering==0:
            sele_ele_index.append(index_row[element_row.index(i)])
        elif octahedral_clustering==1:
            sele_ele_index.append([index_row[element_row.index(i)],get_six_neighbor_atoms(ele_neighbor,element_row.index(i)+1)])
        sele_atom.append(i)
        
#print(sele_ele_index)
if octahedral_clustering==0:
    data1=pd.Series(sele_ele_index)
    #print(data1)
    data=pd.DataFrame(data1)
    data.columns = ['Magnetic']
    print(data)
elif octahedral_clustering==1:

    data=pd.DataFrame(sele_ele_index)
    data.columns = ['Magnetic','Octaheral_vol']
    print(data)
    
    
max_clusters = 10 if len(data)>10 else len(data)



# 加载数据
if os.path.isfile(f"data.save/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl"):
    with open(f"data.save/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    #print(loaded_data)
    inertia_values = loaded_data['inertia_values']
    kmeans_models = loaded_data['kmeans_models']

    # 输出每一步中每个模型包含的点的索引
    for i, kmeans in enumerate(kmeans_models):
        labels = kmeans.labels_
        cluster_indices = {}
        for j in range(len(labels)):
            label = labels[j]
            if label in cluster_indices:
                cluster_indices[label].append(j)
            else:
                cluster_indices[label] = [j]
        #print(f"cluster_number:{i+1}")
        if (i+1) == i_th_pic:
            for label, indices in cluster_indices.items():
                ele_name=[sele_atom[jj] for jj in indices]
                print(f"cluster {label}: {ele_name}")
                print(convert_to_ranges(ele_name))
else:
    inertia_values = []
    kmeans_models = []
    
    # 计算不同簇数下的模型惯性(inertia)值
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
        kmeans_models.append(kmeans)
    
    
    save_data = {}
    save_data[f"kmeans_models"] = kmeans_models
    save_data[f'inertia_values'] = inertia_values
    # 输出每一步中每个模型包含的点的索引
    for i, kmeans in enumerate(kmeans_models):
        labels = kmeans.labels_
        cluster_indices = {}
        for j in range(len(labels)):
            label = labels[j]
            if label in cluster_indices:
                cluster_indices[label].append(j)
            else:
                cluster_indices[label] = [j]
        #print(f"cluster_number:{i+1}")
        if (i+1) == i_th_pic:
            for label, indices in cluster_indices.items():
                ele_name=[sele_atom[jj] for jj in indices]
                print(f"cluster {label}: {ele_name}")
                print(convert_to_ranges(ele_name))
    # 保存数据到文件
    with open(f"data.save/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl", "wb") as file:
        pickle.dump(save_data, file)

#end_time = time.time()
#run_time = end_time - start_time

#print(f"程序运行时间：{run_time} 秒")

# 绘制肘部法则图
plt.figure(figsize=(6, 5))
# 第一张图：肘部法则
#plt.subplot(1, 2, 1)
plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='-')
plt.xlabel('Cluster Number')
plt.ylabel('Inertia Values')
plt.title(f'Elbow Rule of {clustering_element}')
plt.grid(True)
plt.savefig(f"data.save/Elbow_Rule_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
#os.system("python show_fig.py Elbow_Rule.png &")
process1 = subprocess.Popen(["python", "show_fig.py", f"data.save/Elbow_Rule_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png"])
# 第二张图：数据点的分布
#plt.subplot(1, 2, 2)
plt.figure(figsize=(6, 5))
plt.xticks(rotation=45)
labels=[]
for i, kmeans in enumerate(kmeans_models):
    if octahedral_clustering==1:
        labels.append(kmeans.labels_)
    #print(labels,type(labels))
    elif octahedral_clustering==0:
        labels=kmeans.labels_
        plt.scatter(data['Magnetic'], [i] * len(data), c=labels, cmap='viridis')
if octahedral_clustering==1:
    plt.scatter(data['Magnetic'], data['Octaheral_vol'], c=labels[i_th_pic-1], cmap='viridis')
    plt.xlabel('Magnetic')
    plt.ylabel('Octahedral_vol')
    plt.title(f'Data Point Distribution of {clustering_element} k={i_th_pic}')
elif octahedral_clustering==0:  
    plt.xlabel('Magnetic')
    plt.ylabel('Model')
    plt.title(f'Data Point Distribution of {clustering_element}')
plt.grid(True)
#plt.tight_layout()
plt.savefig(f"data.save/Point_distribute_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
#process2 = subprocess.Popen(["python", "show_fig.py", "Point_distribute.png"])
os.system(f"python show_fig.py data.save/Point_distribute_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
#plt.savefig("Elbow_Rule.svgz")
#plt.show()
