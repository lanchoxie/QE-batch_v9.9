import os
import sys

mode=sys.argv[1]
if len(sys.argv)==2:
    file_result="NAN"
else:
    file_result=sys.argv[2]

root_dir = os.path.expandvars('$HOME')
#code_path="%s/bin/QE-batch"%root_dir
code_path=sys.path[0]
def read_files(infiles,split_syb):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "Direct" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(split_syb)
        for i in a_bf:
            if len(i)>0:
                read_data_row.append(i)
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data

depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if ".vasp" in filename:
                str_in_dir.append(filename.split(".vasp")[0].strip())


if file_result=="NAN":    
    for i in str_in_dir:
        if os.path.isfile("%s/replace.py"%code_path)==1:
            os.system("python %s/replace.py %s %s 0 0"%(code_path,i,mode))
        else:
            print("plz\n cp /hpc/data/home/spst/zhengfan/open/replace/replace.py .")



elif file_result!="NAN":
    results=read_files(file_result," ") 
    all_files=[result[0].strip() for result in results]
    job_done=[]
    not_done=[]
    error_file=[]
    for i in results:
        if i[-1].find("DONE")!=-1:
            job_done.append(i[0].strip())
        elif (i[-1].find("waiting")!=-1)|(i[-1].find("running")!=-1):
            not_done.append(i[0].strip())
        else:
            error_file.append(i[0].strip())
#print(job_done,not_done)
    for i in str_in_dir:
        if i.strip() in error_file:
            print("error occur %s %s"%(i,mode))
            if os.path.isfile("%s/replace.py"%code_path)==1:
                os.system("python %s/replace.py %s %s 0 0"%(code_path,i,mode))
            else:
                print("plz\n cp /hpc/data/home/spst/zhengfan/open/replace/replace.py .")
        elif i.strip() in not_done:
            print("not done %s %s"%(i,mode))
        elif i.strip() in job_done:
            print("JOB done %s %s"%(i,mode))
        elif i.strip() not in all_files:
            print("No calculation %s %s"%(i,mode))
            if os.path.isfile("%s/replace.py"%code_path)==1:
                os.system("python %s/replace.py %s %s 0 0"%(code_path,i,mode))
            else:
                print("plz\n cp /hpc/data/home/spst/zhengfan/open/replace/replace.py .")
