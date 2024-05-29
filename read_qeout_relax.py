#!/usr/bin/python2.7
import sys
import os
import math
import re
import string
import numpy
from numpy.linalg import *
#import matplotlib.pyplot as pyplot



###pre-define string operation
def float_all(string):
    floats=[]
    for i in range(0,len(string)):
        floats.append(float(string[i]))
    return floats

def s_string(longstring):
    return float_all(longstring.split( ))

def s_list(longlist):
    lists=[]
    for i in range(0,len(longlist)):
        lists.append(s_string(longlist[i]))
    return lists


def bohr2ang(input): return input/1.889725989

###read_files():
def read_files(filename):
    infile = open(filename, "r")
    raweig = infile.read()
    infile.close()
    return raweig



###find lattice parameter
def find_parameters(qefile_in):

    n=re.compile(r"\s*bravais-lattice index\s+=\s+([\-\d]*)")
    p=re.compile(r"\s*celldm\(1\)=\s+([\d\.\d]+)\s")
    q=re.compile(r"\s*unit-cell volume[\s\w\(\)]+=\s+([\d\.\d]+)\s+\(a\.u\.\)\^3")
    r=re.compile(r"\s*number of atoms\/cell[\s\w\(\)]+=\s+([\d]+)")
    s=re.compile(r"\s*number of electrons[\s\w\(\)]+=\s+([\d\.\d]+)")
    t=re.compile(r"\s*number of Kohn-Sham states=\s+([\d]+)")
    v=re.compile(r"\s*Exchange-correlation=\s+([A-Z\s]+)")
    u=re.compile(r"\s*Program PWSCF (v[\d\.]+)")

    ibrav_index=int(n.findall(qefile_in)[-1] )
    alat=float_all(p.findall(qefile_in))[-1]
    volume=float_all(q.findall(qefile_in))[-1]
    natom=int(r.search(qefile_in).group(1))
    nelec=float(s.search(qefile_in).group(1))
    nbnd=int(t.search(qefile_in).group(1))
    ex=v.search(qefile_in).group(1)
    version=u.search(qefile_in).group(1)
    
    return ibrav_index, alat, volume, natom, nelec, nbnd, ex, version



###find lattice vector
def find_lattice(qefile_in):

    vc_lat_raw=os.popen(f"grep \"CELL_PARA\" -A 4 {sys.argv[1]} | tail -n 4").readlines()
    #print(f"grep \"CELL_PARA\" -A 4 {sys.argv[1]} | tail -n 4")
    if len(vc_lat_raw)!=0:
        vc_lat=[]
        for i in vc_lat_raw:
            if len(i)<2:
                continue
            vc_lat.append([float(vc) for vc in i.strip("\n").strip().split() if len(vc)!=0])
        #print(vc_lat)
        #return vc_lat
        print("VC-relax detected!\n",numpy.array(vc_lat))
        return numpy.array(vc_lat)
    p=re.compile(r"\s*a\(\d\)\s=\s\(([\s\d\.\-]+)\s+\)")
    lattice=numpy.array(s_list(p.findall(qefile_in)))
    print("relax detected!\n",lattice)
    return lattice



###find atomic coordinates
def find_coord(qefile_in,natom):

    name=[];coord=[]

    #p=re.compile(r"\s*\d+\s*([A-Z][a-z]*)\s+tau\(\s*\d+\)\s*=\s*\(([\s\d\.\-]+)\)")
    p=re.compile(r"\s*\d+\s*([A-Z][a-zA-Z]*\d*)\s+tau\(\s*\d+\)\s*=\s*\(([\s\d\.\-]+)\)")
    rawcoord=p.findall(qefile_in)

    #print(rawcoord)
    for i in range(0,len(rawcoord)):
        name.append(rawcoord[i][0])
        coord.append(s_string(rawcoord[i][1]))

    return name[0:natom],numpy.array(coord)

def find_opt_coord(qefile_in):
    coord = []
    p = re.compile(r"ATOMIC_POSITIONS \([a-z]*\)\n(([A-Z][a-z]*\s*[\s\d\-\.]+)+)\n")
    rawcoord = p.findall(qefile_in)
    #print(rawcoord)
    if len(rawcoord)!= 0:
#       print rawcoord[0]
        coord = [ [  float_all(jraw.split( )[1:4] )  for jraw in iraw[0].rstrip('\n').split('\n')] for iraw in rawcoord]
    return coord

def find_opt_coord_unit(qefile_in):
    p = re.compile("ATOMIC_POSITIONS \(([a-z]*)\)\n")
    rawcoord = p.search(qefile_in).group(1)
    if rawcoord.lower() == 'angstrom':
        unit = 'angstrom'
    elif rawcoord.lower() == 'crystal':
        unit = 'crystal'
    else:
        print( 'Unknow unit, use crystal!')
        unit = 'crystal'
    return unit

def find_opt_vect(qefile_in):
    vect = []
    p = re.compile(r"CELL_PARAMETERS \(alat= [\d\.]*\)\n\s*([\d\.\s\-]+)\n " )
    rawvect = p.findall(qefile_in)
    return rawvect
    


###find stresses
def find_stress(qefile_in):

    static_stress=[];non_bar_stress=[];bar_stress=[]
    p=re.compile(r"\s*total   stress  \(Ry\/bohr\*\*3\)\s*\(kbar\)\s*P=([\d\.\-\s]+)\n")
    rawstress=s_list(p.findall(qefile_in))

    for i in range(0, len(rawstress)):
        static_stress.append(rawstress[i][0])
        non_bar_stress.append([]); bar_stress.append([])
        for j in range(1,4):
            non_bar_stress[i].append(rawstress[i][(j-1)*6+1:(j-1)*6+4])
            bar_stress[i].append(rawstress[i][(j-1)*6+4:(j-1)*6+7])
        non_bar_stress[i]=numpy.array(non_bar_stress[i])
        bar_stress[i]=numpy.array(bar_stress[i])

    return  numpy.array(static_stress)[-1],non_bar_stress[-1],bar_stress[-1]



###find energies
def find_energy(qefile_in):

    p=re.compile(r"!\s*total energy\s*=\s*([\d\.\-]+)\s*Ry")
    energy=p.findall(qefile_in)

    if energy==[]:
       print( "Convergence is not achieved ...")
       sys.exit(0)

    return float_all(energy )



###find magnetization
def find_mag(qefile_in):

    p=re.compile(r"\s*total magnetization\s*=\s*([\d\.\-\s]+)\s+Bohr\s+mag\/cell")
    q=re.compile(r"\s*absolute magnetization\s*=\s*([\d\.\-\s]+)\s+Bohr\s+mag\/cell")

    #tot_mag=float_all(p.findall(qefile_in) )
    #abs_mag=float_all(q.findall(qefile_in) )
    tot=(p.findall(qefile_in) )
    abs=(q.findall(qefile_in) )
    
    if tot==[] or abs==[]:
       tot_mag=[0.0];abs_mag=[0.0]
    else:
       tot_mag=s_string(tot[-1])
       abs_mag=s_string(abs[-1])


    return tot_mag, abs_mag


def cal_reciprocal(vect):  return inv(vect).transpose()


def print_xsf(name,vect,coordinates,alat):

   #vect=vect*bohr2ang(alat)
   #coordinates=coordinates*bohr2ang(alat)
    vlen = numpy.shape(vect)[0]

    print ('DIM-GROUP')
    print ('          3           1')
    print ('PRIMVEC')
    for i in range(vlen-3, vlen):
        print ("  %11.7f   %11.7f   %11.7f" % (vect[i,0],vect[i,1],vect[i,2]))
    
    print ('PRIMCOORD')
    print ('         %d           1' %(len(name)))
    for i in range(0, len(name)):
        print ("%s    %11.7f   %11.7f   %11.7f" % (name[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]))

    return

def export_xsf(name,vect,coordinates,alat,filename):

   #vect=vect*bohr2ang(alat)
   #coordinates=coordinates*bohr2ang(alat)
    vlen = numpy.shape(vect)[0]

    f=open(filename,"w")
    f.write ('DIM-GROUP\n' )
    f.write ('          3           1\n' )
    f.write ('PRIMVEC\n' )
    for i in range(vlen-3, vlen):
        f.write( "%11.7f   %11.7f   %11.7f\n" % (vect[i,0],vect[i,1],vect[i,2]) )
    
    f.write ('PRIMCOORD\n' )
    f.write ('         %d           1\n' %(len(name)) )
    for i in range(0, len(name)):
        f.write ("%s    %11.7f   %11.7f   %11.7f\n" % (name[i], coordinates[i,0], coordinates[i,1], coordinates[i,2]) )

    f.close()
    return

readqe_output = read_files(sys.argv[1].strip())
ibrav,alat,volume,natom,nelec,nbnd,ex,version = find_parameters(readqe_output)
lattice_vectors = find_lattice(readqe_output)
name,coordinates = find_coord(readqe_output,natom)
opt_coord = find_opt_coord(readqe_output)
unit = find_opt_coord_unit(readqe_output)
if unit == 'angstrom':
    opt_coord_cart = numpy.copy(opt_coord[-1])
if unit == 'crystal':
    opt_coord_cart = numpy.dot( numpy.array( opt_coord[-1] ), lattice_vectors  )
filename = sys.argv[1]+'.xsf'
export_xsf(name,lattice_vectors,opt_coord_cart,alat,filename)
