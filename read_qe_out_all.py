import os
import sys

depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if ".vasp" in filename:
                str_in_dir.append(filename.split(".vasp")[0])

#print(str_in_dir)
str_in_dir=sorted(str_in_dir,key=len,reverse=True)
report=[]
for i in str_in_dir:
    if os.path.isfile("%s/out_relax_%s"%(i,i))==1:

        f=open("%s/out_relax_%s"%(i,i)).readlines()
        if "JOB DONE" in f[-2]:
            report.append([i,"success"])
            os.system("python /hpc/data/home/spst/zhengfan/open/replace/read_qeout_relax.py %s/out_relax_%s"%(i,i))
        else:
            err=0
            for j in f:
                if "Error" in j:
                    err=1
            if err==0:
                report.append([i,"running"])
            else:
                report.append([i,"Error"])
    else:
        report.append([i,"waiting"])
print("you can use following code to download the files:\nsz ./*/*.xsf\n")
for i in range(len(report)):
    print(report[i])
