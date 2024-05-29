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

#The neighbor element type
O_neighbor=Composition("O")
Li_neighbor=Composition("Li")
ele_neighbor=O_neighbor

df_dir='data.save'
str_dir=f'{str_in}'
#data_file_dir:
df_all_file = f'data.save/bader_charge_of_{str_in}.data'
#str_file_dir:
str_file = f"{str_dir}/out_relax_{str_in}.xsf"

out_file = f"{df_dir}/ATOMIC_INFO_of_{str_in}.txt"

verbose=0

#change the input element into pymatgen readable format
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

def get_nearest_neighbor_dis(neighbor_element_type,atomic_number):
    
    element_type=Composition(neighbor_element_type)
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
        #print(nb_spe[i]) ##what the fuck is this code??
        if ((nb_spe[i]==element_type)&(structure.sites[center_atom_index].species!=element_type)):
            nb_ele_coord.append(nb_coords[i])
            nb_ele_spe.append(nb_spe[i])
            nb_ele_index.append(nb_index[i]+1)
           
    #print(nb_ele_spe)
    #print(nb_ele_coord)
    #print(nb_ele_index)    
    output_nearest=[]
    ele_info=[]
    if len(nb_ele_coord)!=0:
        for i in range(len(nb_ele_coord)):
            #calculate the distance between the neighbors and the center to select the nearest neighbors
            dis=np.linalg.norm(nb_ele_coord[i] - np.array(structure.sites[center_atom_index].coords))
            ele_info.append([nb_ele_spe[i],nb_ele_index[i],nb_ele_coord[i],dis])
        ele_nearest=sorted(ele_info, key=lambda x: x[-1])[0]
        output_nearest=[str(ele_nearest[0])[:-1],ele_nearest[-1]]
    else:
        output_nearest=[neighbor_element_type,99999]
    if verbose==1:
        print(ele_info_min_one)
        
    return "%.6f"%output_nearest[1]

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
        #print(nb_spe[i]) ##what the fuck is this code??
        if ((nb_spe[i]==element_type)&(structure.sites[center_atom_index].species!=element_type))|((structure.sites[center_atom_index].species==element_type)&(nb_spe[i]!=element_type)):
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
    nearest_dis_lst=[x[3] for x in ele_info_min_six]

    #if structure.sites[center_atom_index].coords==O_neighbor:
    #    return 0,0,0,0
    center_coord=np.array(structure.sites[center_atom_index].coords)
    oxygen_coords=nearest_coord_lst.copy()
    #print(atomic_number,structure.sites[center_atom_index].species)
    #print(center_coord,oxygen_coords)
    OF=calculate_distance_variance(center_coord, oxygen_coords)
    angle_deviation_mean=np.mean(calculate_angle_deviations(center_coord, oxygen_coords))
    DI=calculate_distortion_index(center_coord, oxygen_coords)

    #print(nb_site)
    # convert the coord of the neighbors into numpy array
    vertices = np.array(nearest_coord_lst)
    # To calculate the volume of an octahedron, you can use ConvexHull from the Scipy library
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    volume = hull.volume
    #print(f"Calculated the volume of {atomic_number}: {volume}")
    return "%.6f"%volume,"%.6f"%OF,"%.6f"%angle_deviation_mean,"%.6f"%DI

def calculate_octahedral_factor(center_atom, oxygen_atoms): #THIS is a wrong function, unused,reserve for debug
    """
    Calculate the Octahedral Factor for a given set of central atom and oxygen atoms.
    
    Parameters:
    - center_atom: Coordinates of the central atom [x, y, z]
    - oxygen_atoms: List of coordinates for the six oxygen atoms [[x1, y1, z1], [x2, y2, z2], ...]
    
    Returns:
    - Octahedral Factor (OF)
    """
    distances = [np.linalg.norm((center_atom) - (oxy)) for oxy in oxygen_atoms]
    
    d_max = max(distances)
    d_min = min(distances)
    d_ave = np.mean(distances)
    
    OF = (d_max - d_min) / d_ave
    return OF

def calculate_distance_variance(center_atom, oxygen_atoms):
    """
    Calculate the variance of distances between a central atom and a set of oxygen atoms.
    
    Parameters:
    - center_atom: Coordinates of the central atom [x, y, z]
    - oxygen_atoms: List of coordinates for the oxygen atoms [[x1, y1, z1], [x2, y2, z2], ...]
    
    Returns:
    - Variance of the distances
    """
    distances = [np.linalg.norm(np.array(center_atom) - np.array(oxy)) for oxy in oxygen_atoms]
    variance = np.var(distances)

    return variance

def calculate_angle_deviations(center_atom, oxygen_atoms):
    """
    Calculate the deviations of angles from 90 and 180 degrees for a given set of central atom and oxygen atoms.
    
    Parameters:
    - center_atom: Coordinates of the central atom [x, y, z]
    - oxygen_atoms: List of coordinates for the six oxygen atoms [[x1, y1, z1], [x2, y2, z2], ...]
    
    Returns:
    - List of angle deviations
    """
    angles = []
    for i in range(6):
        vec1 = (oxygen_atoms[i]) - (center_atom)
        for j in range(i+1, 6):
            vec2 = (oxygen_atoms[j]) - (center_atom)
            if vec1.shape != (3,) or vec2.shape != (3,):
                raise ValueError(f"Unexpected shape for vectors at indices {i} and {j}. vec1 shape: {vec1.shape}, vec2 shape: {vec2.shape}")
            cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(cosine_angle))
            
            # Calculate deviation from 90 or 180
            deviation = min(abs(angle - 90), abs(angle - 180))
            angles.append(deviation)
    
    return angles



def calculate_distortion_index(center_atom, surrounding_atoms):
    """
    Calculate the Distortion Index (DI) for a given set of central atom and surrounding atoms.
    
    Parameters:
    - center_atom: Coordinates of the central atom [x, y, z]
    - surrounding_atoms: List of coordinates for the surrounding atoms [[x1, y1, z1], [x2, y2, z2], ...]
    
    Returns:
    - Distortion Index (DI)
    """
    distances = [np.linalg.norm((center_atom) - (atom)) for atom in surrounding_atoms]
    d_ave = np.mean(distances)
    DI = np.sqrt(np.mean([(d - d_ave)**2 for d in distances]))
    
    return DI


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

str_read_in1=open(infile).readlines()
str_read_in2=""
for i in str_read_in1:
    str_read_in2+=i
xsf = XSF_1.from_string(str_read_in2)
structure = xsf.structure
element_row=[]
for i in structure:
    element_row.append([i.species,structure.index(i)+1])
output_data=[]
for i,v in enumerate(element_row):
    #if v[0]!=O_neighbor:
        #octa_data_i,OF_out,ADM_out,DI_out=get_six_neighbor_atoms(ele_neighbor,v[1])
    #else:
        #octa_data_i=0
    print(f"{i}/{len(element_row)}","\r",end="")
    octa_data_i,OF_out,ADM_out,DI_out=get_six_neighbor_atoms(ele_neighbor,v[1])
    Ti_nd=get_nearest_neighbor_dis("Ti",v[1])
    Co_nd=get_nearest_neighbor_dis("Co",v[1])
    Ni_nd=get_nearest_neighbor_dis("Ni",v[1])
    Mn_nd=get_nearest_neighbor_dis("Mn",v[1])
    output_data.append(f"{v[1]}-{str(v[0])[:-1]}\t{octa_data_i}\t{OF_out}\t{ADM_out}\t{DI_out}\t{Ti_nd}\t{Co_nd}\t{Ni_nd}\t{Mn_nd}")

f_out=open(out_file,"w+")
f_out.writelines("#atom_name\t八面体体积\t八面体键长方差\t平均角度畸变\t畸变因子DI\t最近邻Ti距离\t最近邻Co距离\t最近邻Ni距离\t最近邻Mn距离\n")
for i in output_data:
    #print(i)
    f_out.writelines(i+"\n")
print(out_file,"created!")
