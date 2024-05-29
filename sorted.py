import os
import sys

file_read=sys.argv[1]
sort_column=int(sys.argv[2])

def read_files_str(infiles,split_syb):
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
                read_data_row.append(str(i))
        read_data.append(read_data_row)
   # print(len(read_data),len(read_data[0]))
    f.close()
    return read_data 

data_read=read_files_str(file_read," ")

data_read=sorted(data_read,key=lambda x:x[sort_column])
for i in data_read:
    for j in i:
        print(j,"  ",end="")
    print()
