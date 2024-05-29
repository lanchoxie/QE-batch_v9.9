import os
import sys

command=sys.argv[1]
judge_file=sys.argv[2]

str_in_dir0=[]
depth=1
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if ".vasp" in filename:
                str_in_dir0.append(filename.split(".vasp")[0])

print(judge_file)
print(str_in_dir0)
for i,v in enumerate(str_in_dir0):
    print(v)
    print(judge_file)
    judge_file_i=judge_file.replace("*",v)
    print(judge_file_i)
    if not os.path.isfile(judge_file_i):
        print(f"python {command} {v}")
        jj=os.popen(f"python {command} {v}").readlines()
    else:
        a=judge_file_i.replace("*",v)
        print(f"{a} FOUND!!")
