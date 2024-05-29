import numpy
import os
import time
import sys


error_waiting_recal=1  #set this on to recal for error files
wavefc_rm=0   #!!!!!!DON‘T set to 1 if you want further calculation like BANDS!!!!
              #set this to 1 to remove wavefunction when job is done
prefix="BTO"  #the prefix same in your input file

qe_dir="/hpc/data/home/spst/zhengfan/open/replace"
root_dir = os.path.expandvars('$HOME')
code_dir="%s/bin/QE-batch"%root_dir
############job_script_settings##########

flat_info=open("%s/flat.save/FLAT_INFO"%code_dir).readlines()
for lines in flat_info:
    if lines.find("flat_number=")!=-1:
        flat_number=int(lines.split("flat_number=")[-1].split("#")[0])
    if lines.find("node_num=")!=-1:
        node_num=int(lines.split("node_num=")[-1].split("#")[0])
    if lines.find("ppn_num")!=-1:
        ppn_num_man=int(lines.split("ppn_num=")[-1].split("#")[0])
        if ppn_num_man==0:
            ppn_set=0
        elif ppn_num_man!=0:
            ppn_set=1
    if lines.find("wall_time=")!=-1:
        wall_time=lines.split("wall_time=")[-1].split("#")[0].strip("\n")


#flat_number=2           # 1 for"zf_normal",2 for "spst_pub",3 for "dm_pub_cpu"
#node_num=2
#wall_time="116:00:00"
#ppn_set=1
#ppn_num_man=16
####################"zf_normal" has 52ppn,"spst_pub" has 32 ppn ,"dm_pub_cpu" has 32 ppn
if flat_number==1:
    type_flat="zf_normal"
if flat_number==2:
    type_flat="spst_pub"
if flat_number==3:
    type_flat="dm_pub_cpu"
###################

def JOB_func(type_flat,node_num):

    type_hpc=""
    if type_flat=="spst_pub":
        ppn_num=32
        type_hpc="pbs"
    elif type_flat=="zf_normal":
        ppn_num=52
        type_hpc="pbs"
    elif type_flat=="dm_pub_cpu":
        ppn_num=32
        type_hpc="sbatch"
    if ppn_set==1:
       ppn_num=ppn_num_man

    ppn_tot=node_num*ppn_num

    if type_hpc=="pbs":
        jobscript_file_in=['#!/bin/bash\n',
        '#PBS  -N   coolxty\n',
        '#PBS  -l   nodes=%d:ppn=%d\n'%(node_num,ppn_num),
        '#PBS  -l   walltime=%s\n'%wall_time,
        '#PBS  -S   /bin/bash\n',
        '#PBS  -j   oe\n', 
        '#PBS  -q   %s\n'%(type_flat),
        '\n',
        'cd $PBS_O_WORKDIR\n',
        '\n',
        'NPROC=`wc -l < $PBS_NODEFILE`\n',
        'JOBID=$( echo $PBS_JOBID | awk -F \".\"  \'{print $1}\')\n',
        "echo \'JOB-ID = \'  $JOBID >> JOB-${JOBID}\n",
        "echo This job has allocated $NPROC procs >> JOB-${JOBID}\n",
        "TW=$( qstat -f \"$JOBID\" | grep -e \'Resource_List.walltime\' | awk -F \"=\" \'{print $(NF)}\' )\n",
        "echo \'requested walltime = \'  $TW  >> JOB-${JOBID}\n",
        "Q=$( qstat -f \"$JOBID\" | grep -e \'queue\' | awk \'{print $3}\' )\n",
        "echo \'requested queue = \'  $Q >> JOB-${JOBID}\n",
        "TC=$( qstat -f \"$JOBID\" | grep -e \'etime\' | awk -F \"=\" \'{print $(NF)}\' )\n",
        "TC_s=$( date -d \"$TC\" +%s )\n",
        "echo \'time submit = \'  $TC $TC_s >> JOB-${JOBID}\n",
        "TS=$( qstat -f \"$JOBID\" | grep -e \'start_time\' | awk -F \"=\" \'{print $(NF)}\' )\n",
        "TS_s=$( date -d \"$TS\" +%s )\n",
        "echo \'time start = \'  $TS  $TS_s >> JOB-${JOBID}\n",
        "TimeDu=`expr $TS_s - $TC_s`\n",
        "echo \'Waiting time(s) = \'  $TimeDu  >> JOB-${JOBID}\n",
        "\n",
        "echo This job has allocated $NPROC proc > log\n",
        "\n",
        "module load compiler/intel/2021.3.0\n",
        "module load mpi/intelmpi/2021.3.0\n",
        "\n",
        ]
    if type_hpc=="sbatch":
        jobscript_file_in=[
        "#!/bin/bash\n",
        "#SBATCH --job-name=xty\n",
        "#SBATCH -D ./\n",
        "#SBATCH --nodes=%d\n"%node_num,
        "#SBATCH --ntasks-per-node=%d\n"%ppn_num,
        "#SBATCH -o output.%j\n",
        "##SBATCH -e error.%j\n",
        "#SBATCH --time=%s\n"%wall_time,
        "#SBATCH --partition=dm_pub_cpu\n",
        "\n",
        "##SBATCH --gres=gpu:4 #if use gpu, uncomment this\n",
        "#export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so\n",
        "ulimit -s unlimited\n",
        "ulimit -l unlimited\n",
        "\n",
        "#setup intel oneapi environment \n",
        "source /dm_data/apps/intel/oneapi/setvars.sh\n",
        "#source /etc/profile\n",
        "module load compiler/latest\n",
        "module load mpi/latest\n",
        "module load mkl/latest\n",
        ]
    return jobscript_file_in,type_hpc,ppn_tot

def JOB_modify(JOB,mode,molecule_i,type_hpc_out,ppn_tot_out):
    vaspfile=molecule_i
    fj=open(JOB,"w+")
    jobscript_file_1=[]
    for i in range(len(jobscript_file)):
        j_rename=""
        #print(jobscript_file[i],type(jobscript_file[i]),i)
        if type_hpc_out=="pbs":
            if jobscript_file[i].find('#PBS  -N')!=-1:
                job_name_split=vaspfile.split("_")
                for j_names in job_name_split:
                    j_rename+=j_names[:2].zfill(2)
                jobscript_file[i]='#PBS  -N   %s\n'%(j_rename)
            jobscript_file_1.append(jobscript_file[i])
        if type_hpc_out=="sbatch":
            if jobscript_file[i].find('#SBATCH --job-name')!=-1:
                job_name_split=vaspfile.split("_")
                for j_names in job_name_split:
                    j_rename+=j_names[:2].zfill(2)
                jobscript_file[i]='#SBATCH --job-name=%s\n'%(j_rename)
            jobscript_file_1.append(jobscript_file[i])
    if type_hpc_out=="pbs":
        if solvation_model==0:
            jobscript_file_1.append("mpirun --bind-to core -np $NPROC -hostfile $PBS_NODEFILE %s/pw-6.8.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(qe_dir,mode,molecule_i,mode,molecule_i)) 
        elif solvation_model==1:
            jobscript_file_1.append("mpirun --bind-to core -np $NPROC -hostfile $PBS_NODEFILE %s/pw-7.2-environ.x -npool 4 -ndiag 4 --environ < in_%s_%s  >& out_%s_%s"%(qe_dir,mode,molecule_i,mode,molecule_i))
    if type_hpc_out=="sbatch":
        if solvation_model==0:
            jobscript_file_1.append("mpirun --bind-to core -np %s %s/pw-6.8.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))
        elif solvation_model==1:
            jobscript_file_1.append("mpirun --bind-to core -np %s -hostfile $PBS_NODEFILE %s/pw-7.2-environ.x -npool 4 -ndiag 4 --environ < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))


    fj.writelines(jobscript_file_1)
    jobscript_file_1=[]
    fj.close()



jobscript_file,type_hpc_out,ppn_tot_out=JOB_func(type_flat,node_num)
##default value of mode is relax
if len(sys.argv)==1:
    mode="relax"
else:
    mode=sys.argv[1]
print(mode)

dir_file=(os.popen("pwd").read())
dir=max(dir_file.split('\n'))
JOB_name=""
error=0
if os.path.isfile("%s/replace.py"%code_dir)==1:
    replace_line=open("%s/replace.py"%code_dir).readlines()
#get the job_script name:
    for i in replace_line:
        if i.find("JOB=")!=-1:
            JOB_name=i.split("JOB=")[-1].split("#")[0].split("\"")[1]
            #print("JOB_name**************",JOB_name)
        if i.find("#solvation")!=-1:
            solvation_model=int(i.split("#")[0].split("solvation_model=")[-1].strip())
    if (JOB_name==""):
        print("ERROR: replace.py not set correctly!")
        error=1


elif os.path.isfile("%s/replace.py"%code_dir)==0:
    print("plz copy /hpc/data/home/spst/zhengfan/open/replace/replace.py here, otherwise some func does not work! ")
    error=1

sub_method=""
if (flat_number==1)|(flat_number==2):
    sub_method="qsub"
elif (flat_number==3):
    sub_method="sbatch"


#print("**************************%s %s"%(sub_method,JOB_name))

find_bcon=0
root_dir = os.path.expandvars('$HOME')
#print("%s/bin/bcon.sh"%root_dir)
if os.path.isfile("%s/bin/bcon.sh"%root_dir)==1:
    print("bcon.sh found")
    find_bcon=1
if find_bcon==0:
    print("bcon.sh not found,copy from /hpc/data/home/spst/zhengfan/open/ to ~/bin")
    os.system("cp /hpc/data/home/spst/zhengfan/open/bcon.sh  ~/bin/.")

if error==0:    
    str_in_dir=[]
    str_in_dir0=[]
    depth=1
    stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
    for root,dirnames,filenames in os.walk(stuff):
        if root[len(stuff):].count(os.sep) < depth:
            for filename in filenames:
                if ".vasp" in filename:
                    str_in_dir0.append(filename.split(".vasp")[0])
        for dirname in dirnames:
            if dirname in str_in_dir0:
                str_in_dir.append(dirname)
    str_in_dir=list(set(str_in_dir))
    str_in_dir=sorted(str_in_dir,key=len,reverse=True)
    print("*******************",str_in_dir)
    len_max=len(str_in_dir[0])
    #print(str_name0)
    fileout=[]
    yes_to_all_continue=""
    yes_to_all_error=""
    yes_to_all_error1=""
    for i in range(len(str_in_dir)):
        if os.path.isfile("%s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i])))==1:
            etot1=(os.popen("grep JOB %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines())
            
            if len(etot1)!=1:
                etot0=(os.popen("grep \"! \" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i])))).readlines()
                read_error=0
                etot11=open("%s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()
                for error_line in  etot11:
                    if error_line.find("BAD TERMINATION")!=-1:
                        read_error=2
                    if error_line.find("ERROR")!=-1:
                        read_error=2
                    if error_line.find("stopping ...")!=-1:
                        read_error=2
                    if error_line.find("DUE TO TIME LIMIT")!=-1:
                        read_error=1
                    if error_line.find("error waiting for event")!=-1:
                        read_error=2
                    if error_line.find("longjmp causes")!=-1:
                        read_error=1
                    if error_line.find("process killed")!=-1:
                        read_error=1
                    if error_line.find("CANCELLED AT")!=-1:
                        read_error=2
                    if error_line.find("error parsing")!=-1:
                        read_error=2
                if (read_error==1):
                    if len(etot0) > 0:
                        if etot0[0].find("failed")==-1:
                            energy0=etot0[-1].split(" =")[-1].split("Ry")[0].strip()
                            #print("energy0",energy0,"11",etot0[-1])
                            energy_eV0=float(energy0)*27.211396/2
                            print("%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    out of timing")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    out of timing"])
                        elif etot0[0].find("failed")!=-1:
                            print("%*s"%(len_max,str_in_dir[i]),"unknown error")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),"unknown error"])
                    elif len(etot0)==0:
                        print("%*s"%(len_max,str_in_dir[i]),"out of timing")
                        fileout.append(["%*s"%(len_max,str_in_dir[i]),"out of timing"])
                #recalculate if out of timing
                    if (yes_to_all_error==""):
                        print("would you like me to continue cal for you?")
                        print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                        continue_error=input(">>>>>>")
                        if continue_error.lower()=="y":
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            #a=os.popen("pwd").readlines()
                            #print(a)
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                                print("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_error.lower()=="q":
                            yes_to_all_error="q"
                        elif continue_error.lower()=="a":
                            yes_to_all_error="a"
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                            
                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_error.lower()=="n":
                            print("skip this file")
                        else:
                            print("unknow input,skip this loop")
                    elif (yes_to_all_error=="a"):
                                                                    
                        os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                        if mode=="relax":
                            os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

                        JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                        os.system("%s %s"%(sub_method,JOB_name))
                        os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                        os.chdir(dir)
                        print("%s subed!"%str_in_dir[i])
                    
                if (read_error==2):
                    if len(etot0) > 0:
                        if etot0[0].find("failed")==-1:
                            energy0=etot0[-1].split(" =")[-1].split("Ry")[0].strip()
                            #print("energy0",energy0,"11",etot0[-1])
                            energy_eV0=float(energy0)*27.211396/2
                            print("%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    ERROR")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    ERROR"])
                        elif etot0[0].find("failed")!=-1:
                            print("%*s"%(len_max,str_in_dir[i]),"unknown error")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),"unknown error"])
                    elif len(etot0)==0:
                        print("%*s"%(len_max,str_in_dir[i]),"ERROR")
                        fileout.append(["%*s"%(len_max,str_in_dir[i]),"ERROR"])
                #recalculate if out of timing
                    if (yes_to_all_error==""):
                        print("would you like me to continue cal for you?")
                        print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                        continue_error=input(">>>>>>")
                        if continue_error.lower()=="y":
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            #a=os.popen("pwd").readlines()
                            #print(a)
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                                print("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_error.lower()=="q":
                            yes_to_all_error="q"
                        elif continue_error.lower()=="a":
                            yes_to_all_error="a"
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                            
                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_error.lower()=="n":
                            print("skip this file")
                        else:
                            print("unknow input,skip this loop")
                    elif (yes_to_all_error=="a"):
                                                                    
                        os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                        if mode=="relax":
                            os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

                        JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                        os.system("%s %s"%(sub_method,JOB_name))
                        os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                        os.chdir(dir)
                        print("%s subed!"%str_in_dir[i])
                #output some data if there is no error
                elif read_error==0:
                    if len(etot0) > 0:
                        energy0=etot0[-1].split(" =")[-1].split("Ry")[0].strip()
                        energy_eV0=float(energy0)*27.211396/2
                        print("%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    running",end='')
                        if len(os.popen("grep \"Gradient error\" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines())>0:
                            force_now=(os.popen("grep \"Gradient error\" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()[-1]).split("Gradient error")[-1].strip().split("=")[-1].strip()
                            print("   ",force_now)
                        else:
                            print("force not generated!")
                        fileout.append(["%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    running"])
                    elif len(etot0)==0:
                        print("%*s"%(len_max,str_in_dir[i]),"running")
                        fileout.append(["%*s"%(len_max,str_in_dir[i]),"running"])
            #if there is JOB DONE output:
            else:
                if mode=="relax":
                    etot_relax_0=os.popen("grep ! %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()
                    if len(etot_relax_0)==0:#if no energy data output
                        print("!!!!!!!!!!",str_in_dir[i],"scf_not_converge")
                        etot="a = 0 Ry"
                    else:
                        etot=etot_relax_0[-1] 
                elif mode=="bands":
                    print("%*s"%(len_max,str_in_dir[i]),"BANDS","    !!DONE!!!")
                    if wavefc_rm==1:
                        if os.path.exists("%s/%s.save"%(str_in_dir[i],prefix))==1:
                            os.system("rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            print("!!rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.mix*"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.mix*"%(str_in_dir[i],prefix))
                    fileout.append(["%*s"%(len_max,str_in_dir[i]),"BANDS","    !!DONE!!!"])
                    continue
                elif mode=="scf":
                    etot=(os.popen("grep ! %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()[0])
                energy=etot.split(" =")[-1].split("Ry")[0].strip()
                energy_eV=float(energy)*27.211396/2

                lines=open("%s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()
                if "JOB" in lines[-2]:#relax/scf converged
                    if wavefc_rm==1:
                        if os.path.exists("%s/%s.save"%(str_in_dir[i],prefix))==1:
                            os.system("rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            print("!!rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.mix*"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.mix*"%(str_in_dir[i],prefix))
                    print("%*s"%(len_max,str_in_dir[i]),energy,"Ry","%.11f"%energy_eV,"eV","    !!DONE!!!")
                    fileout.append(["%*s"%(len_max,str_in_dir[i]),energy,"Ry","%.11f"%energy_eV,"eV","    !!DONE!!!"])
    
                if "JOB" not in lines[-2]:#relax/scf not converged
                    print("%*s"%(len_max,str_in_dir[i]),energy,"Ry","%.11f"%energy_eV,"eV","    not converged")
                    fileout.append(["%*s"%(len_max,str_in_dir[i]),energy,"Ry","%.11f"%energy_eV,"eV","    not converged"])
                    if (yes_to_all_continue==""):
                        print("would you like me to continue cal for you?")
                        print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                        continue_cal=input(">>>>>>")
                        if continue_cal.lower()=="y":
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            #a=os.popen("pwd").readlines()
                            #print(a)
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                                print("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_cal.lower()=="q":
                            yes_to_all_continue="q"
                        elif continue_cal.lower()=="a":
                            yes_to_all_continue="a"
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            if mode=="relax":
                                os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                            JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                            os.system("%s %s"%(sub_method,JOB_name))
                            os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))                            
                            os.chdir(dir)
                            print("%s subed!"%str_in_dir[i])
                        elif continue_cal.lower()=="n":
                            print("skip this file")
                        else:
                            print("unknow input,skip this loop")
                    elif (yes_to_all_continue=="a"):
                        os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                        if mode=="relax":
                            os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                        JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                        os.system("%s %s"%(sub_method,JOB_name))
                        os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                        os.chdir(dir)
                        print("%s subed!"%str_in_dir[i])
        else:
            #print("I am here~")
            #error exists recalculate
            job_on_hpc=os.popen("python %s/job_on_hpc.py %s"%(code_dir,str_in_dir[i])).readlines()[0].strip("\n")
            if job_on_hpc=="waiting on line": 
                print("%*s"%(len_max,str_in_dir[i]),"waiting")
                fileout.append(["%*s"%(len_max,str_in_dir[i]),"waiting"])
            elif job_on_hpc=="Not here":
                print("%*s"%(len_max,str_in_dir[i]),"No Calculation")
                fileout.append(["%*s"%(len_max,str_in_dir[i]),"No Calculation"])
            elif job_on_hpc=="running":
                print("%*s"%(len_max,str_in_dir[i]),"running")
                fileout.append(["%*s"%(len_max,str_in_dir[i]),"running"])
            if (yes_to_all_error1=="")&(error_waiting_recal==1)&(job_on_hpc=="Not here"):
                print("would you like me to restart cal for you?")
                print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                continue_error1=input(">>>>>>")
                if continue_error1.lower()=="y":
                    os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                    #a=os.popen("pwd").readlines()
                    #print(a)
                    if mode=="relax":
                        os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                        print("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                    JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                    os.system("%s %s"%(sub_method,JOB_name))
                    os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                    os.chdir(dir)
                    print("%s subed!"%str_in_dir[i])
                elif continue_error1.lower()=="q":
                    yes_to_all_error1="q"
                elif continue_error1.lower()=="a":
                    yes_to_all_error1="a"
                    os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                    if mode=="relax":
                        os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                    JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                    os.system("%s %s"%(sub_method,JOB_name))
                    os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                    os.chdir(dir)
                    print("%s subed!"%str_in_dir[i])
                elif continue_error1.lower()=="n":
                    print("skip this file")
                else:
                    print("unknow input,skip this loop")
            elif (yes_to_all_error1=="a"):
                os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                if mode=="relax":
                    os.system("bcon.sh out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                os.system("%s %s"%(sub_method,JOB_name))
                os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                os.chdir(dir)
                print("%s subed!"%str_in_dir[i])
    f_ou=open("pv_result_out","w+")
    for i in range(len(fileout)):
        for j in range(len(fileout[i])):
            f_ou.write(str(fileout[i][j])+"\t")
        f_ou.write("\n")
    print("pv_result_out generated!")
