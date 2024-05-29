import sys
import os

dir_read=sys.argv[1]
mode=sys.argv[2]

file_in=f"in_{mode}_{dir_read}"
file_out=f"out_{mode}_{dir_read}"

atomic_number=0

f_in_buffer=open(f"{dir_read}/{file_in}").readlines()
f_out_buffer=open(f"{dir_read}/{file_out}").readlines()


read_start=0
read_count=1
atomic_species=[]
for i,v in enumerate(f_in_buffer):
    if "nat" in v:
        atomic_number=int(v.split("=")[-1].strip("\n").strip(",").strip())
    if 'ATOMIC_POSITIONS' in v:
        read_start=1
        continue
    if (read_start==1)&(read_count<atomic_number+1):
        read_count+=1
        atomic_species.append([x for x in v.split() if len(x)>0][0]) 
        
if atomic_number==0:
    raise ValueError(f"No nat found in input file of {dir_read}/{file_in}!!!")

mark_index=[]
mag_data=[]
for i,v in enumerate(f_out_buffer):
    if "Magnetic moment per site" in v:
        mark_index.append(i)
for i in range(max(mark_index)+1,max(mark_index)+atomic_number+1):
    mag_data.append(f_out_buffer[i])

#print(atomic_species[0],atomic_species[-1],mag_data[0],mag_data[-1])
#print(atomic_number,len(atomic_species),len(mag_data))

fnm_out=f"Mag_of_{dir_read}.txt"
f_output=open(fnm_out,"w+")
for i in range(len(mag_data)):
    f_output.writelines(f"{atomic_species[i]}    {mag_data[i]}")
print(f"{fnm_out} generated!")
