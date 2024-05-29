import os
import sys

input_file=sys.argv[1]
output_file=sys.argv[2]

input_cont=open(input_file).readlines()
atom_number=0
for line in input_cont:
    if "nat" in line and "!" not in line:
        atom_number=int(line.strip("\n").split(",")[0].split("=")[1].strip())
if atom_number==0:
    raise ValueError("No nat found in inputfile!")
output_cont=os.popen(f"grep 'ATOMIC' -A {atom_number} {output_file}").readlines()
#for i in output_cont:
#    print(i)

lattice_raw=os.popen(f"grep CELL_PARAMETERS -A 3 {input_file}").readlines()
#for i in lattice:
#    print(i)
def split_lines(infiles,split_syb):
    f1=infiles
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "Direct" in lines:
            continue
        if "***" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(split_syb)
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(str(i))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    return read_data

def rl_aw(element_i):
    element=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Uun","Uuu","Uub"]   #end in 112
    return element.index(element_i)+1

atom_species=[rl_aw(i[0]) for i in split_lines(output_cont[1:atom_number+1]," ")]    #储存所有原子的序号（重复）
#print(len(atom_species),atom_species)
lattice_vec=split_lines(lattice_raw[1:]," ")
str_count=0
read_start=0
read_count=0
str_coord=[]
read_time="0"

output_cont_out_seg=split_lines(output_cont," ")
for i,v in enumerate(output_cont_out_seg):
    #print(output_cont_out_seg[i][0])
    if v[0].find("ATOMIC")==-1 and v[0].find("--")==-1:
        output_cont_out_seg[i][0]=str(rl_aw(output_cont_out_seg[i][0]))
    elif v[0].find("--")!=-1:
        pass
    else:
        output_cont_out_seg[i][0]="ATOMIC"
#output_cont_out=["  "+"    ".join(i)+"\n" for i in output_cont_out_seg]
output_cont_out=[i for i in output_cont_out_seg]
#for i in output_cont_out:
#    print(i)

for i in output_cont_out:
    if i[0].find("ATOMIC")!=-1: 
        #read_time=i.split("=")[-1].strip("\n").strip()
        read_start=1
        str_coord.extend([
  " %d atoms,Iteration =   %d image=   %d, Etot =  -0.5425748649E+05, Average Force=  0.54877E+00, Max force=  0.15096E+01\n"%(atom_number,(str_count+1)/5,(str_count+1)%5),
  #" %s atoms,TimeStep =   %s, \n"%(str(str_atom_number),read_time),
 " Lattice vector\n",
 "   %.10e    %.10e    %.10e\n"%(float(lattice_vec[0][0]),float(lattice_vec[0][1]),float(lattice_vec[0][2])),
 "   %.10e    %.10e    %.10e\n"%(float(lattice_vec[1][0]),float(lattice_vec[1][1]),float(lattice_vec[1][2])),
 "   %.10e    %.10e    %.10e\n"%(float(lattice_vec[2][0]),float(lattice_vec[2][1]),float(lattice_vec[2][2])),
 " Position, move_x, move_y, move_z\n"])
        #str_coord.append("Direct configuration=          %d\n"%(str_count+1))
        continue
    if read_start==1:
        buffer_coord=[]
        buffer_coord.append("  %2d "%int(i[0]))
        buffer_coord.extend("    %.8f    %.8f    %.8f     "%(float(i[1]),float(i[2]),float(i[3])))
        buffer_coord.extend("0  0  0\n")
        str_coord.append(buffer_coord)
        #str_coord.append(i)
        read_count+=1
    if read_count==atom_number:
        str_coord.append("Force\n")
        for j in range(len(atom_species)):
            str_coord.append(" %d 0 0 0\n"%atom_species[j])
        str_coord.append([" -------------------------------------------------\n"])
        read_start=0
        read_count=0
        str_count+=1
        continue
f2=open("MOVEMENT","w+")
for i in range(len(str_coord)):
    f2.writelines(str_coord[i])
print("MOVEMENT created!")
