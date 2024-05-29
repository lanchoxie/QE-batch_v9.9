import os
import sys
import re

acf="ACF.dat"
infile=sys.argv[1]
if os.path.isfile(f"in_scf_{infile}"):
    atomic_file=f"in_scf_{infile}"
elif os.path.isfile(f"in_relax_{infile}"):
    atomic_file=f"in_relax_{infile}"
else:
    raise ValueError("No scf or relaxation found!")
bader_file=f"Bader_{infile}"

in_spe=open(atomic_file).readlines()
in_charge=open(acf).readlines()
in_bader=open(bader_file).readlines()

atomic_number=0
for i,v in enumerate(in_spe):
    if 'nat' in v and '!' not in v:
        atomic_number=int(v.split("=")[-1].strip("\n").strip().strip(",").strip())
if atomic_number==0:
    raise ValueError(f"No \'nat\' found in {atomic_file}")
elements=[]
start_read_in_spe=0
read_count_in_spe=0
for i,v in enumerate(in_spe):
    if "ATOMIC_POSITIONS" in v:
        start_read_in_spe=1
        continue
    if (start_read_in_spe==1)&(read_count_in_spe<atomic_number):
        elements.append([x for x in v.split() if len(x)>0][0])

start_read_in_bader=0
bader_charge_lst=[]
for i,v in enumerate(in_charge):
    if "#" in v:
        continue
    if (start_read_in_bader==0)&(v.find("-------------")!=-1):
        start_read_in_bader=1
        continue
    if (start_read_in_bader==1)&(v.find("-------------")!=-1):
        start_read_in_bader=0
        continue
    if  start_read_in_bader==1:
        bader_charge_lst.append(float([x for x in v.strip("\n").split() if len(x)>0][4]))

valence_charge=[]
for i,v in enumerate(in_bader):
    #print(v)
    pattern=r'^\s+\d+\s+[A-Za-z0-9]+\s+\d+(?:\.\d*)?\s*\n?$'
    if re.match(pattern,v):
        valence_line=[x for x in v.strip("\n").split() if len(x)>0]
        print(valence_line)
        valence_charge.append([valence_line[1],float(valence_line[2])])

print(valence_charge)

ionic_charge=[]
for i,v in enumerate(elements):
    if v in [x[0] for x in valence_charge]:
        ionic_charge.append("%.6f"%(valence_charge[[x[0] for x in valence_charge].index(v)][1]-bader_charge_lst[i]))

for i in range(len(ionic_charge)):
    print(f"{elements[i]}-{i+1}    {bader_charge_lst[i]}    {ionic_charge[i]}")           
