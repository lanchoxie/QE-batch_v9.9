import os
import sys 
import re

str_names=sys.argv[1]

dir_file=(os.popen("pwd").read())
Current_dir=max(dir_file.split('\n'))
root_dir = os.path.expandvars('$HOME')
#code_path="%s/bin/QE-batch/"%root_dir
code_path=sys.path[0]


def bader_process(str_name):
    file_input_tag=[]
    for root,dirs,files in os.walk(Current_dir+"/"+str_name):
        for file in files:
            if "in_scf" in file or "in_relax" in file:
                file_input_tag.append(file)
    if len(file_input_tag)==0:
        raise ValueError("No input found!The input shall be named with \'in_scf\' or '\in_relax\'!")
    spin_mode=0
    input_tag=open(str_name+"/"+file_input_tag[0]).readlines()
    for lines in input_tag:
        if "nspin" in lines and "2" in lines and "!" not in lines:
            spin_mode=1

    infile=str_name
    if (os.path.isfile(f"data.save/bader_charge_of_{infile}.data")==1)&(os.path.isfile(f"data.save/Lowdin_of_{infile}.txt")==1):
        if ((spin_mode==1)&(os.path.isfile(f"data.save/Mag_of_{infile}.txt")==1))|(spin_mode==0):
            print(f"Quick read charge of {infile}")
            bader_file=f"{str_name}/Bader_{infile}"
            in_bader=open(bader_file).readlines()
            valence_charge=[]
            for i,v in enumerate(in_bader):
                #print(v)
                pattern=r'^\s+\d+\s+[A-Za-z0-9]+\s+\d+(?:\.\d*)?\s*\n?$'
                if re.match(pattern,v):
                    valence_line=[x for x in v.strip("\n").split() if len(x)>0]
                    #print(valence_line)
                    valence_charge.append([valence_line[1],float(valence_line[2])])
            outshit=''
            for i in valence_charge:
                outshit+=f"{i[0]}-{[i[1]]}\n"
            return outshit
    os.chdir(str_name)
    jj=os.popen(f"bader Bader_{str_name}.cube").readlines()
    print(f"bader Bader_{str_name}.cube")
    os.chdir(Current_dir)


    magfile=f"data.save/Mag_of_{str_name}.txt"
    lowdin_file=f"data.save/Lowdin_of_{str_name}.txt"

    if os.path.isfile(f"{str_name}/in_scf_{infile}"):
        print(f"scf done in {str_name}")
        atomic_file=f"{str_name}/in_scf_{infile}"
        if not os.path.isfile(magfile) and (spin_mode==1):
            read_mag=os.popen(f"python {code_path}/read_mag.py {str_name} scf").readlines()
            os.system(f"mv Mag_of_{str_name}.txt data.save")
        if not os.path.isfile(lowdin_file):
            if os.path.isfile(f"{str_name}/out_pdos_{str_name}"):
                read_mag=os.popen(f"python {code_path}/read_lowdin.py {str_name} scf").readlines()
                os.system(f"mv Lowdin_of_{str_name}.txt data.save")
            else:
                raise ValueError("PDOS has not been calculated in {str_name}!")
    elif os.path.isfile(f"{str_name}/in_relax_{infile}"):
        print(f"relax done in {str_name}")
        atomic_file=f"{str_name}/in_relax_{infile}"
        if not os.path.isfile(magfile) and (spin_mode==1):
            read_mag=os.popen(f"python {code_path}/read_mag.py {str_name} relax").readlines()
            os.system(f"mv Mag_of_{str_name}.txt data.save")
        if not os.path.isfile(lowdin_file):
            if os.path.isfile(f"{str_name}/out_pdos_{str_name}"):
                read_mag=os.popen(f"python {code_path}/read_lowdin.py {str_name} relax").readlines()
                os.system(f"mv Lowdin_of_{str_name}.txt data.save")
            else:
                raise ValueError("PDOS has not been calculated in {str_name}!")
    else:
        raise ValueError("No scf or relaxation found!")

    bader_file=f"{str_name}/Bader_{infile}"
    acf=f"{str_name}/ACF.dat"
    if spin_mode==1:
        in_mag=open(magfile).readlines()     #Magnetic file get by read_mag.py
    in_lowdin=open(lowdin_file).readlines()
    in_spe=open(atomic_file).readlines()    
    in_charge=open(acf).readlines()      #bader charge file get by bader 
    in_bader=open(bader_file).readlines() #bader charge file get by pp.x calculation
    
    atomic_number=0
    element_number=0
    for i,v in enumerate(in_spe):
        if 'nat' in v and '!' not in v:
            atomic_number=int(v.split("=")[-1].strip("\n").strip().strip(",").strip())
        if 'ntyp' in v and '!' not in v:
            element_number=int(v.split("=")[-1].strip("\n").strip().strip(",").strip())
    if (atomic_number==0)|(element_number==0):
        raise ValueError(f"No \'nat\' or \'ntyp\' found in {atomic_file}")
    elements=[]             #read elements from in_scf/in_relax it is elements related to EACH atomic number
    start_read_in_spe=0
    read_count_in_spe=0
    for i,v in enumerate(in_spe):
        if "ATOMIC_POSITIONS" in v:
            start_read_in_spe=1
            continue
        if (start_read_in_spe==1)&(read_count_in_spe<atomic_number):
            elements.append([x for x in v.split() if len(x)>0][0])

    element_type=[]             #read element_type from in_scf/in_relax it is element with no duplicates
    start_read_ele=0
    read_count_ele=0
    for i,v in enumerate(in_spe):
        if "ATOMIC_SPECIES" in v:
            start_read_ele=1
            continue
        if (start_read_ele==1)&(read_count_ele<element_number):
            element_type.append([x for x in v.split() if len(x)>0][0])

    element_type_number=[]
    for i in elements:
        if i in [x[0] for x in element_type_number]:
            element_type_number[[x[0] for x in element_type_number].index(i)][1]+=1
        else:
            element_type_number.append([i,1])        

    start_read_in_bader=0
    bader_charge_lst=[]   #bader charge of each atomic number
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
    
    valence_charge=[]  #total charge read from Bader_{str_name} which gets from pesuedo potentials
    for i,v in enumerate(in_bader):
        #print(v)
        pattern=r'^\s+\d+\s+[A-Za-z0-9]+\s+\d+(?:\.\d*)?\s*\n?$'
        if re.match(pattern,v):
            valence_line=[x for x in v.strip("\n").split() if len(x)>0]
            #print(valence_line)
            valence_charge.append([valence_line[1],float(valence_line[2])])
    for i,v in enumerate(valence_charge):  #replace the element incase the spin up and spin down is not required by Bader_{infile} file
        v[0]=element_type[i]        

    total_charge=0
    for i in valence_charge:
        print(i[0],i[1],element_type_number[[x[0] for x in element_type_number].index(i[0])][0],element_type_number[[x[0] for x in element_type_number].index(i[0])][1])
        total_charge+=i[1]*element_type_number[[x[0] for x in element_type_number].index(i[0])][1]
    print("Calculated total charge:",total_charge)

    lowdin_charge=[]     #lowdin charge read from out_pdos
    for i,v in enumerate(in_lowdin):
        if "#" in v:
            continue
        else:
            lowdin_line=[x for x in v.strip("\n").split("\t") if len(x)>0]
            lowdin_charge.append([float(lowdin_line[1])])

    lowdin_sum=0
    for i in lowdin_charge:
        lowdin_sum+=i[0]

    for i in lowdin_charge:
        i[0]=float("%.6f"%(i[0]*total_charge/lowdin_sum))

    if spin_mode==1:
        magnetics=[]     #magnetics read from out_scf/out_relax
        for i,v in enumerate(in_mag):
            mag_line=[x for x in v.strip("\n").split(" ") if len(x)>0]
            magnetics.append([float(mag_line[-1])])

    ionic_charge=[]      #calculated by valence_charge-bader/lowdin charge
    #print(valence_charge,lowdin_charge)
    for i,v in enumerate(elements):
        if v in [x[0] for x in valence_charge]:
            ionic_charge.append(["%.6f"%(valence_charge[[x[0] for x in valence_charge].index(v)][1]-bader_charge_lst[i]),
                                 "%.6f"%(valence_charge[[x[0] for x in valence_charge].index(v)][1]-lowdin_charge[i][0]) ] )
    
    
    out_filenm=f"bader_charge_of_{infile}.data"
    f_out=open(out_filenm,"w")
    for i in range(len(ionic_charge)):
        if spin_mode==1:
            f_out.writelines(f"{elements[i]}-{i+1}    {lowdin_charge[i][0]}    {bader_charge_lst[i]}    {ionic_charge[i][1]}    {ionic_charge[i][0]}    {magnetics[i][0]}\n")           
        elif spin_mode==0:
            f_out.writelines(f"{elements[i]}-{i+1}    {lowdin_charge[i][0]}    {bader_charge_lst[i]}    {ionic_charge[i][1]}    {ionic_charge[i][0]}\n")           
        #print(f"{elements[i]}    {bader_charge_lst[i]}    {ionic_charge[i]}")           
    f_out.close()
    os.system(f"mv {out_filenm} data.save")
    outshit=''
    for i in valence_charge:
        outshit+=f"{i[0]}-{[i[1]]}\n"
    return outshit

bader_process(str_names)
