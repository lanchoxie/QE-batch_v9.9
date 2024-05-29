
import sys, os
import re
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pyplot

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("BandsOut", help="output of QE bands calculation", type=str)
parser.add_argument("NumofVal", help="number of valence bands", type=int)
parser.add_argument("EneBlFermi", help="showing range of energy below Fermi (positive number)", type=float)
parser.add_argument("EneAbFermi", help="showing range of energy above Fermi (positive number)", type=float)
parser.add_argument("-t","--title", help="addition string append to output figure", type=str)
parser.add_argument("-f","--fermi", help="input fermi energy, otherwise automatically set it to bandgap middle", type=float)
args = parser.parse_args()


# USAGE:

# ./titledbandstrv5.1.py M1.bands.out num_valence_bands  energy_below_Efermi(ev)  energy_above_Efermi(ev)  title(str)  input_Fermi(eV)
# eg: python titledbandstrv5.1.py M1.bands.out 40  4.0  4.0  -t 'gga' -f  11.032

def float_all(string):
    floats=[]
    for i in range(0,len(string)):
        floats.append(float(string[i]))
    return floats


def read_bands(filename):
    infile = open(filename, "r")
    raweig = infile.read()
    infile.close()
    p=re.compile(r"k[\w ]*=([\- ]*\d\.\d{4})([\- ]*\d\.\d{4})([\- ]*\d\.\d{4})([\w\d\(\)\s\:]*)([\d\s\.\-]*)",re.M)
    rawkpts= p.findall(raweig)
    p=re.compile(r"[\- \d][\d]{1,3}\.\d+")
    kpts=[]
    for ikpt in range(0, len(rawkpts)):
        kpts.append([numpy.array([float(rawkpts[ikpt][0]),float(rawkpts[ikpt][1]),float(rawkpts[ikpt][2])]),float_all(p.findall(rawkpts[ikpt][4]))])
    return kpts



def create_plot_array(kpts):


    kline=[0.0]
    klabel=[[kpts[0][0]],[kline[0]]]
    kdiff=numpy.zeros(3,dtype=float)
    
    bands=[]
    for i in range(0,len(kpts[0][1])):
        bands.append([kpts[0][1][i]])
    
    for ikpt in range(1,len(kpts)):
        kdiff_old=kdiff
        kdiff=kpts[ikpt][0]-kpts[ikpt-1][0]
        
        if (numpy.vdot(kdiff_old,kdiff )/(numpy.vdot(kdiff,kdiff)*numpy.vdot(kdiff_old,kdiff_old))**0.5<.9):
            klabel[0].append(kpts[ikpt-1][0])
            klabel[1].append(kline[ikpt-1])
    
        kline.append(kline[ikpt-1]+(numpy.vdot(kdiff,kdiff))**0.5)
    
        for i in range(0,len(kpts[ikpt][1])):
            bands[i].append(kpts[ikpt][1][i])
    
    klabel[0].append(kpts[-1][0])
    klabel[1].append(kline[-1])

    return [klabel, kline, bands]

def plot_bandstruct(bandstruct,vband):
    print (vband)
    pyplot.figure(1)
    for i in range(0,vband):
        pyplot.plot(bandstruct[1],bandstruct[2][i], color='black',linewidth=2.0)

    for i in range(vband,len(bandstruct[2])):
        pyplot.plot(bandstruct[1],bandstruct[2][i], color='black',linewidth=1.4)

    if args.fermi:
        fermi = args.fermi
    else:
        fermi=(max(bandstruct[2][vband-1])+min(bandstruct[2][vband]))/2.0
    bandgap=abs(min(bandstruct[2][vband])-max(bandstruct[2][vband-1]))
    
    energies=[[],[]]

    energy_range=5.0
    evrange=[args.EneBlFermi,args.EneAbFermi]
    for i in range(-1*int(evrange[0]),int(evrange[1])+1):
        einterval=float(i)
        energies[0].append(fermi+einterval)
        energies[1].append(' % 02.f'%(einterval))
    print ('Fermi energy = ', fermi)
    print ('Band gap = ', bandgap)
    pyplot.vlines(bandstruct[0][1],fermi-evrange[0],fermi+evrange[1],color='k',linewidth=0.7)
    pyplot.hlines(energies[0],bandstruct[0][1][0],bandstruct[0][1][-1],color='k',linewidth=0.2)
    pyplot.axis([bandstruct[1][0],bandstruct[1][-1],fermi-evrange[0],fermi+evrange[1]])
    pyplot.xticks(bandstruct[0][1],[r'$\Gamma$',r'$\rm{X}$',r'$\rm{M}$',
                                    r'$\Gamma$', r'$\rm{R}$', r'$\rm{X}$',
                                    r'$\Gamma$',r'$Z$'],fontsize=32)
    pyplot.ylabel('$\mathbf{(E-E_{F}) / eV}$',fontsize=38)
    pyplot.yticks(energies[0],energies[1],fontsize=32)
#   pyplot.grid(False, linestyle='dotted')
    pyplot.figure(1).set_size_inches(12, 8)
    regexp = re.compile(r"\d+")    
    pyplot.title('')

    if args.title:
        FileNamePre = args.BandsOut + args.title
    else:
        FileNamePre = args.BandsOut
    pyplot.savefig(FileNamePre+".png")
    pyplot.savefig(FileNamePre+".svgz")
#   filename = sys.argv[1]+".eps" 
#   pyplot.savefig(filename)

def k_labels(klabel):
#    sympoints = { '  0.000  0.000  0.000' : r'$\Gamma$',
#                    '  0.500  0.000  0.000' : r'$\rm{L}$',
#                    '  0.000  0.500  0.500' : r'$\rm{X}$',
#                    '  0.500  0.500  0.500' : r'$\rm{Z}$',
#                    '  0.375  0.375  0.750' : r'$\rm{K}$',
#                    '  0.333  0.333  0.000' : r'$\rm{M}$' }
    sympoints = { '  0.000  0.000  0.000' : r'$\Gamma$',
                  '  0.000  0.500  0.000' : r'$\rm{X}$',
                  '  0.500  0.500  0.000' : r'$\rm{F}$',
                  '  0.613  0.354  0.374' : r'$\rm{M}$',
                  '  0.000  0.000  0.561' : r'$\rm{T}$',
                  '  0.613 -0.354  0.187' : r'$\rm{L}$',
                  '  0.000  0.500  0.500' : r'$\rm{R}$',
                  '  0.500  0.000  0.000' : r'$\rm{C}$' }
#    sympoints = { '  0.000  0.000  0.000' : r'$\Gamma$',
#                  '  0.000  0.500  0.000' : r'$\rm{X}$',
#                  '  0.500  0.500  0.000' : r'$\rm{M}$',
#                  '  0.000  0.000  0.000' : r'$\Gamma$',
#                  '  0.000  0.000  0.500' : r'$\rm{Z}$',
#                  '  0.000  0.500  0.500' : r'$\rm{R}$',
#                  '  0.500  0.500  0.500' : r'$\rm{A}$' }
#    sympoints = { '  0.000  0.000  0.000' : r'$\Gamma$',
#                    '  0.000  0.577  0.000' : r'$\rm{L}$',
#                    '  0.000  0.500  0.500' : r'$\rm{X}$',
#                    '  0.500  0.500  0.500' : r'$\rm{Z}$',
#                    '  0.375  0.375  0.750' : r'$\rm{K}$',
#                    '  0.333  0.577  0.000' : r'$\rm{K}$' }

    symlabels=[]
    print (klabel)
    for k in klabel:
        symlabels.append(sympoints.get(kstring(k)))
        print (kstring(k))
        
    return symlabels

def kstring(k):
    kstr=' % 04.3f'%(k[0]) + ' % 04.3f'%(k[1])+' % 04.3f'%(k[2])

    return kstr    

plot_bandstruct(create_plot_array(read_bands(args.BandsOut)),args.NumofVal)
