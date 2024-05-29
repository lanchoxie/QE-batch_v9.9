import os 
import sys

run_script_dir=sys.argv[1].strip("/")
mode=sys.argv[2]
code_dir=sys.path[0]

dirs_in=[]
depth=1
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(f"{run_script_dir}/.")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for dirname in dirnames:
            dirs_in.append(dirname)

#print(dirs_in)

av_mode=["scf","relax","pdos","bands","BaderCharge"]
if mode not in av_mode:
    raise ValueError(f"{mode} not in {av_mode}!")

out_words=[]
for i in dirs_in:
    print("cp -r {QE-batch,in_relax,SPIN,DFT-U} %s/%s/."%(run_script_dir,i),f"{dirs_in.index(i)+1}/{len(dirs_in)}","\r",end="")
    os.system("cp -r {QE-batch,in_relax,SPIN,DFT-U} %s/%s/."%(run_script_dir,i))
    out_words.append([i,mode])
print()
f=open(f"{code_dir}/time_schedule.py").readlines()
for i,v in enumerate(f):
    if "#make sure the QE-batch software package was named containing \"QE-batch\" string" in v:
        f.insert(i+1,"time_schedule="+str(out_words)+"\n")
f1=open(f"{run_script_dir}/time_schedule-{run_script_dir}.py","w+")
for i in f:
    f1.writelines(i)
f1.close()
print(f"{run_script_dir}/time_schedule-{run_script_dir}.py created!")
