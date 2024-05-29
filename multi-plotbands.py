import sys
import os
import time

verbosity=1

starting_point=int(sys.argv[1])
ending_point=int(sys.argv[2])

dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))


def bands_read(file_bands):
    bands_in_scf=[]
    readed=0
    f=open(file_bands,"r").readlines()
    for i in range(len(f)):
        if "occupation numbers" in f[i]:
            if readed==0:
                #print(i)
                for j in range(i+1,len(f)):
                    bands_line=f[j].split()
                    if len(bands_line)==0:
                        readed=1
                        break
                    elif len(bands_line)>0:
                        for k in bands_line:
                            if float(k)!=0:
                                bands_in_scf.append(k)
    bands_in_valence=int(len(bands_in_scf))
    return bands_in_valence

depth=1
str_in_dir=[]
stuff = os.path.abspath(os.path.expanduser(os.path.expandvars(".")))
for root,dirnames,filenames in os.walk(stuff):
    if root[len(stuff):].count(os.sep) < depth:
        for filename in filenames:
            if ".vasp" in filename:
                str_in_dir.append(filename.split(".vasp")[0])
str_in_dir=list(set(str_in_dir))
str_in_dir=sorted(str_in_dir,key=len,reverse=True)
print("Starting reading bands...")
datas=[]
for i in str_in_dir:
    Fermi_E=0
    Band_gap=0
    print("%d/%d"%(str_in_dir.index(i),len(str_in_dir)),"\r",end='')
    time.sleep(0.01)
    os.chdir("%s/%s"%(current_dir,i))
    vb_num=bands_read("out_scf_"+i)
    jj=os.popen("python /hpc/data/home/spst/zhengfan/open/replace/titledbandstrv5.1.py out_bands_%s %d %d %d"%(i,vb_num,starting_point,ending_point)).readlines()
    os.chdir(current_dir)
    for j_i in jj:
        if j_i.find("Fermi energy =")!=-1:
            Fermi_E=float(j_i.split("\n")[0].split("Fermi energy = ")[-1].strip())
        elif j_i.find("Band gap =")!=-1:
            Band_gap=float(j_i.split("\n")[0].split("Band gap =")[-1].strip())
    datas.append([i,Fermi_E,Band_gap])


f_w=open("electron_str_in_files","w+")
length_li=[len(x[0]) for x in datas]
length_max=max(length_li)
if verbosity==1:
    for data in range(len(datas)):
        for data_i in range(len(datas[data])):
            if data_i!=0:
                print("%.4f"%datas[data][data_i]+"\t",end='')
            if data_i==0:
                print("%*s"%(length_max,datas[data][data_i])+"\t",end='')
        print("\n") 

for data in range(len(datas)):
    for data_i in range(len(datas[data])):
        if data_i!=0:
            f_w.writelines("%.4f"%datas[data][data_i]+"\t")
        if data_i==0:
            f_w.writelines("%*s"%(length_max,datas[data][data_i])+"\t")
    f_w.writelines("\n") 
print("bands readed done!")
print("electron_str_in_files gengerated!")
