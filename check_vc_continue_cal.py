import sys
import os
from libs.modify_file import modify_parameter
import argparse

#input dir / filename parameter read
parser=argparse.ArgumentParser(description="Modify &electron setting to converge in check_vc.py. You must use it with QE-batch")
parser.add_argument('str_name', type=str, help='structure name,i.e. the name of .vasp file')
args=parser.parse_args()

str_name=args.str_name.strip()
#modify the input of pw.x in the input filename 
modify_parameter(str_name, str_name, 'mixing_beta', "0.15", '&electrons','    ')
modify_parameter(str_name, str_name, 'conv_thr', "1.0d-8", '&electrons','    ')

#move the output into a new formatt in input dir to make the files calculate again, since the check_vc.py count the output file number and stop calculation till the output file reach a maximum number, so we need to change the output file name.
out_history_relax_list=[f for f in os.listdir(str_name) if "out_history_relax" in f]
out_history_relax_max=max([int(x.split(".")[-1].strip()) for x in out_history_relax_list]) if len(out_history_relax_list)>0 else 0
out_relax_list=[f for f in os.listdir(str_name) if "out_relax" in f]
out_relax_max=max([int(x.split(".")[-1].strip()) for x in out_relax_list if "." in x])+1 if len(out_relax_list)>0 else 0

for i in out_relax_list:
    if i.find(".")!=-1:
        count=int(i.split(".")[-1].strip())
        new_name=i.replace("out_relax","out_history_relax").replace(f".{count}",f".{count+out_history_relax_max}")
    else:
        new_name=i.replace("out_relax","out_history_relax")+f".{out_relax_max+out_history_relax_max}"
    os.system(f"mv {str_name}/{i} {str_name}/{new_name}")
print("his_max:",out_history_relax_max)
print("relax_max:",out_relax_max)
