import os 
import sys
import argparse

parser = argparse.ArgumentParser(description="This is a scripts for job creation")

parser.add_argument('file_name', type=str, help='Calculated filename')
parser.add_argument('mode', type=str, help='Calculation mode,support: relax,scf,bands,pdos,BaderCharge')
parser.add_argument('--check_flat', '-c', action='store_true', help='Check the flat type, do not create jobscript, and return pbs or slurm')

args = parser.parse_args()

#args:

file_name=args.file_name
mode=args.mode

prefix="BTO"  #the prefix same in your input file

root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
code_dir=sys.path[0]
queue_name='spst'

if os.path.isfile("%s/replace.py"%code_dir)==1:
    replace_line=open("%s/replace.py"%code_dir).readlines()
#get the job_script name:
    for i in replace_line:
        if i.find("JOB=")!=-1:
            JOB=i.split("JOB=")[-1].split("#")[0].split("\"")[1]
            #print("JOB_name**************",JOB_name)
        if i.find("#solvation")!=-1:
            solvation_model=int(i.split("#")[0].split("solvation_model=")[-1].strip())
        if i.find("qe_dir")!=-1:
            qe_dir=(i.split("qe_dir=")[-1].split("#")[0].strip().strip('\"'))
        if i.find("qe_version")!=-1:
            qe_version=(i.split("qe_version=")[-1].strip())
    if (JOB==""):
        print("ERROR: replace.py not set correctly!")
        error=1
elif os.path.isfile("%s/replace.py"%code_dir)==0:
    print("plz copy /hpc/data/home/spst/zhengfan/open/replace/replace.py here, otherwise some func does not work! ")



############job_script_settings##########
###You can modify the FLAT_SAVE file in QE-batch/flat.save/ to manually add default settings of your flats
def read_files(infiles):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "#" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(" ")
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(i)
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data

flat_save=read_files("%s/flat.save/FLAT_SAVE"%code_dir)
#flat save has files format like below:
#1  zf_normal  pbs  52
#flat_number  queue_name  type_flat  ppn_num_defaut

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
#ppn_num=16
#wall_time="116:00:00"
####################"zf_normal" has 52ppn,"spst_pub" has 32 ppn ,"dm_pub_cpu" has 32 ppn

def JOB_func(flat_number):
    flat_ind=[int(i[0]) for i in flat_save].index(flat_number)
    type_flat=flat_save[flat_ind][1]
    type_hpc=flat_save[flat_ind][2]
    ppn_num=int(flat_save[flat_ind][3])
    if ppn_set==1:
        ppn_num=ppn_num_man
    ppn_tot=node_num*ppn_num

    #print(type_flat,node_num,ppn_num,wall_time)
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
    if type_hpc=="slurm":
        jobscript_file_in=[
        "#!/bin/bash\n",
        "#SBATCH --job-name=xty\n",
        "#SBATCH -D ./\n",
        "#SBATCH --nodes=%d\n"%node_num,
        "#SBATCH --ntasks-per-node=%d\n"%ppn_num,
        "#SBATCH -o output.%j\n",
        "##SBATCH -e error.%j\n",
        "#SBATCH --time=%s\n"%wall_time,
        "#SBATCH --partition=%s\n"%(type_flat),
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

def JOB_modify(JOB,mode,molecule_i,type_hpc_out,ppn_tot_out,jobscript_file):
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
        if type_hpc_out=="slurm":
            if jobscript_file[i].find('#SBATCH --job-name')!=-1:
                job_name_split=vaspfile.split("_")
                for j_names in job_name_split:
                    j_rename+=j_names[:2].zfill(2)
                jobscript_file[i]='#SBATCH --job-name=%s\n'%(j_rename)
            jobscript_file_1.append(jobscript_file[i])
    if (type_hpc_out=="pbs") & (mode!="pdos") & (mode!="BaderCharge"):
        if solvation_model==0:
            jobscript_file_1.append("mpirun --bind-to core -np $NPROC -hostfile $PBS_NODEFILE %s/pw-6.8.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(qe_dir,mode,molecule_i,mode,molecule_i)) 
        elif solvation_model==1:
            jobscript_file_1.append("mpirun --bind-to core -np $NPROC -hostfile $PBS_NODEFILE %s/pw-7.2-environ.x -npool 4 -ndiag 4 --environ < in_%s_%s  >& out_%s_%s"%(qe_dir,mode,molecule_i,mode,molecule_i))
    elif (type_hpc_out=="slurm") & (mode!="pdos") & (mode!="BaderCharge"):
        if solvation_model==0:
            jobscript_file_1.append("mpirun --bind-to core -np %s %s/pw-6.8.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))
        elif solvation_model==1:
            jobscript_file_1.append("mpirun --bind-to core -np %s -hostfile $PBS_NODEFILE %s/pw-7.2-environ.x -npool 4 -ndiag 4 --environ < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))
    elif mode=="pdos":
        jobscript_file_1.append("mpirun --bind-to core -np %s %s/projwfc.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))
    elif mode=="BaderCharge":
        jobscript_file_1.append("mpirun --bind-to core -np %s %s/pp.x -npool 4 -ndiag 4 < in_%s_%s  >& out_%s_%s"%(ppn_tot_out,qe_dir,mode,molecule_i,mode,molecule_i))

    fj.writelines(jobscript_file_1)
    jobscript_file_1=[]
    fj.close()

jobscript_file,type_hpc_out,ppn_tot_out=JOB_func(flat_number)

if os.path.isfile(JOB)==1:                                      #job file modify
    os.system("rm %s"%JOB)
os.mknod(JOB)
print(f"###{type_hpc_out}")
if not args.check_flat:
    JOB_modify(JOB,mode,file_name,type_hpc_out,ppn_tot_out,jobscript_file)
    print(f"{JOB} created!")
