# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:18:23 2023

@author: xiety
"""


from __future__ import annotations

from pymatgen.core.periodic_table import Element
# Import modules
from pymatgen.io.xcrysden import XSF
from pymatgen.io.vasp.inputs import Poscar
import os

file='Desktop\\rho\\files_qeout'
dir_=os.path.join(os.path.expanduser("~"), file)

# Get all xsf files in current directory
xsf_files = [f for f in os.listdir(dir_+"/"+".") if f.endswith(".xsf")]
#print(xsf_files)
# Create a new directory named "vasp_files"
if os.path.exists(dir_+"/"+"vasp_files")==True:
    a=1
else:
    os.mkdir(dir_+"/"+"vasp_files")

# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

"""
Support for reading XCrysDen files.
"""


__author__ = "Matteo Giantomassi"
__copyright__ = "Copyright 2013, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Matteo Giantomassi"


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
                    Z = Element(tokens[0]).Z if tokens[0].isalpha() else int(tokens[0])
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





# Loop over xsf files and convert them to vasp files
for xsf_file in xsf_files:
    print(dir_+"/"+xsf_file)
    # Read xsf file and get structure
    f1=open(dir_+"/"+xsf_file).readlines()
    f2=""
    for i in f1:
        f2+=i
    xsf = XSF_1.from_string(f2)

    structure = xsf.structure

    # Delete unwanted strings from file name
    xsf_file = xsf_file.replace("out_relax_", "")
    xsf_file = xsf_file.replace("out_relax_out_relax_", "")

    # Write structure to vasp file in new directory
    vasp_file = xsf_file.replace(".xsf", ".vasp")
    vasp_file = os.path.join(dir_+"/"+"vasp_files", vasp_file)
    poscar = Poscar(structure)
    poscar.write_file(vasp_file)
