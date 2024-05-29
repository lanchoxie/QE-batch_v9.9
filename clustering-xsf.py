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
from sklearn.metrics import silhouette_score
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


i_th_pic=int(sys.argv[3]) #0 for best clusters. seek the clustering picture when set octahedral_clustering=1
octahedral_clustering=int(sys.argv[4])  #set this to 1 to introduce octahedral volume

print_fig=0
#str_in='LiNiO2_331_NC_21'
#clustering_element="Ni"
#i_th_pic=4

df_dir='data.save'
str_dir=f'{str_in}'
#data_file_dir:
df_all_file = f'data.save/bader_charge_of_{str_in}.data'
#str_file_dir:
str_file = f"{str_dir}/out_relax_{str_in}.xsf"

verbose=0

#change the input element into pymatgen readable format
Clu_ele_str=Composition(clustering_element)
infile=str_file

code_dir=sys.path[0]

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
        lst_data.append(int("".join([j for j in i.split("-")[-1] if j.isdigit()])))
    #print(lst_data)
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



# 加载数据
if os.path.isfile(f"{df_dir}/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl"):
    with open(f"{df_dir}/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    #print(loaded_data)
    inertia_values = loaded_data['inertia_values']
    kmeans_models = loaded_data['kmeans_models']
    sele_atom = loaded_data[f'sele_atom']
    max_clusters = loaded_data[f'max_clusters']
    data = loaded_data[f'data']

    #silhouette coefficient:https://blog.csdn.net/qq_19672707/article/details/106857918
    #轮廓系数（Silhouette Coefficient）是一种用于度量聚类质量的指标，它结合了簇内点的紧密度（Cohesion）和簇间点的分离度（Separation）。轮廓系数的范围通常在[-1, 1]之间，具体定义如下：
    #对于每个样本点，计算它与同簇内所有其他点的平均距离，记为a（紧密度）。
    #对于该样本点，计算它与最近的一个不同簇内的所有点的平均距离，记为b（分离度）。
    #轮廓系数S被定义为：S = (b - a) / max(a, b)
    #轮廓系数的解释如下：
    #
    #如果S接近1，表示样本点距离其同簇内的其他点很远，同时与其他簇的点距离较近，表示聚类效果很好。
    #如果S接近0，表示样本点距离同簇内的其他点和其他簇的点都差不多，这种情况下聚类效果较差。
    #如果S接近-1，表示样本点距离同簇内的其他点很近，但与其他簇的点距离较远，表示样本点可能被错误地分配到了不正确的簇中。
    #因此，轮廓系数越接近1表示聚类效果越好，越接近-1表示聚类效果较差。通常，选择具有最高轮廓系数的聚类数量作为最佳聚类数量。
    
    #需要注意的是，轮廓系数在某些情况下可能存在局限性，特别是在密度不均匀的数据集上。因此，它应该与其他评估指标一起使用，以更全面地评估聚类质量
    best_score = -1  # 初始化最佳轮廓系数得分
    best_n_clusters = 1  # 初始化最佳聚类数量
    for n_clusters in range(2,max_clusters):#Valid values are 2 to n_samples - 1 (inclusive)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        # 打印当前聚类数量的轮廓系数
        print(f"聚类数量={n_clusters}, 轮廓系数={silhouette_avg}")
        
        # 如果当前轮廓系数比之前的最佳分数高，更新最佳分数和最佳聚类数量
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters

    # 打印最佳的聚类数量
    print(f"最佳聚类数量={best_n_clusters}, 最佳轮廓系数={best_score}")
    print("*********************************************************")
    print("*******You can use below code in sum_qe_dos_diy.py*******\n")
    for i, kmeans in enumerate(kmeans_models):
        labels = kmeans.labels_
        cluster_indices = {}
        for j in range(len(labels)):
            label = labels[j]
            if label in cluster_indices:
                cluster_indices[label].append(j)
            else:
                cluster_indices[label] = [j]
        best_n_clusters=best_n_clusters if i_th_pic==0 else i_th_pic
        if (i+1) == best_n_clusters:
            for label, indices in cluster_indices.items():
                ele_name=[int(sele_atom[jj].split("-")[-1]) for jj in indices]
                print(f"###cluster {label}: {ele_name}")
                #print("{"+f"{convert_to_ranges(ele_name)}"+"}-tot")



else:
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
    
    df_all = pd.read_csv(df_all_file, sep='\s+',header=None)
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
        #if i.split("-")[0] not in element_spe:
            #print(i)
            #element_spe.append(i.split("-")[0])
        #print(i)
        if ''.join([s for s in i.split("-")[0] if not s.isdigit()]) not in element_spe:
            #print(i)
            element_spe.append(''.join([s for s in i.split("-")[0] if not s.isdigit()]))
            
    print(element_spe)
    if clustering_element in element_spe:
        pass
    elif clustering_element not in element_spe:
        raise ValueError(f"{clustering_element} not in {element_spe}")
    
    for i in element_row:
        if (''.join([s for s in i.split("-")[0] if not s.isdigit()])) == clustering_element:
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
    save_data[f'sele_atom'] = sele_atom
    save_data[f'max_clusters'] = max_clusters
    save_data[f'data'] = data


    # 保存数据到文件
    with open(f"{df_dir}/{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.pkl", "wb") as file:
        pickle.dump(save_data, file)

    best_score = -1  # 初始化最佳轮廓系数得分
    best_n_clusters = 1  # 初始化最佳聚类数量
    for n_clusters in range(2, max_clusters):#Valid values are 2 to n_samples - 1 (inclusive)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(data)
        
        # 计算轮廓系数
        silhouette_avg = silhouette_score(data, cluster_labels)
        
        # 打印当前聚类数量的轮廓系数
        print(f"聚类数量={n_clusters}, 轮廓系数={silhouette_avg}")
        
        # 如果当前轮廓系数比之前的最佳分数高，更新最佳分数和最佳聚类数量
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
    
    # 打印最佳的聚类数量
    print(f"最佳聚类数量={best_n_clusters}, 最佳轮廓系数={best_score}")
    print("*********************************************************")
    print("*******You can use below code in sum_qe_dos_diy.py*******\n")
    for i, kmeans in enumerate(kmeans_models):
        labels = kmeans.labels_
        cluster_indices = {}
        for j in range(len(labels)):
            label = labels[j]
            if label in cluster_indices:
                cluster_indices[label].append(j)
            else:
                cluster_indices[label] = [j]
        best_n_clusters=best_n_clusters if i_th_pic==0 else i_th_pic
        if (i+1) == best_n_clusters:
            for label, indices in cluster_indices.items():
                ele_name=[sele_atom[jj] for jj in indices]
                #print(f"cluster {label}: {ele_name}")
                #print("{"+f"{convert_to_ranges(ele_name)}"+"}-tot")
                ele_name=[int(sele_atom[jj].split("-")[-1]) for jj in indices]
                print(f"###cluster {label}: {ele_name}")

#end_time = time.time()
#run_time = end_time - start_time

#print(f"程序运行时间：{run_time} 秒")
print("*********************************************************")
print("*********************************************************")

if print_fig !=0:
    # 绘制肘部法则图
    plt.figure(figsize=(6, 5))
    # 第一张图：肘部法则
    #plt.subplot(1, 2, 1)
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='-')
    plt.xlabel('Cluster Number')
    plt.ylabel('Inertia Values')
    plt.title(f'Elbow Rule of {clustering_element}')
    plt.grid(True)
    plt.savefig(f"{df_dir}/Elbow_Rule_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
    #os.system("python show_fig.py Elbow_Rule.png &")
    process1 = subprocess.Popen(["python", f"{code_dir}/show_fig.py", f"{df_dir}/Elbow_Rule_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png"])
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
        plt.scatter(data['Magnetic'], data['Octaheral_vol'], c=labels[best_n_clusters-1], cmap='viridis')
        plt.xlabel('Magnetic')
        plt.ylabel('Octahedral_vol')
        plt.title(f'Data Point Distribution of {clustering_element} k={best_n_clusters}')
    elif octahedral_clustering==0:  
        plt.xlabel('Magnetic')
        plt.ylabel('Model')
        plt.title(f'Data Point Distribution of {clustering_element}')
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig(f"{df_dir}/Point_distribute_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
    #process2 = subprocess.Popen(["python", "show_fig.py", "Point_distribute.png"])
    os.system(f"python {code_dir}/show_fig.py {df_dir}/Point_distribute_{str_in}_{clustering_element}_NB-{str(ele_neighbor)}_{octahedral_clustering}.png")
    #plt.savefig("Elbow_Rule.svgz")
    #plt.show()
