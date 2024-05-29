import numpy
import os
import time
import sys


error_waiting_recal=0  #set this on to recal for error files
wavefc_rm=0   #!!!!!!DON‘T set to 1 if you want further calculation like BANDS!!!!
              #set this to 1 to remove wavefunction when job is done
prefix="BTO"  #the prefix same in your input file

qe_dir="/hpc/data/home/spst/zhengfan/open/replace"
root_dir = os.path.expandvars('$HOME')
#code_dir="%s/bin/QE-batch"%root_dir
code_dir=sys.path[0]

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


#print("**************************%s %s"%(sub_method,JOB_name))

find_bcon=0
root_dir = os.path.expandvars('$HOME')
#print("%s/bin/bcon.sh"%root_dir)
if os.path.isfile(f"{code_dir}/bcon.py")==1:
    print("bcon.py found")
    find_bcon=1
if find_bcon==0:
    print("bcon.py not found,plz download the full package of QE-batch!")

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
    if len(str_in_dir)==0:
        print("!!!!!!! No .vasp file found in this directory  !!!")
    len_max=len(str_in_dir[0])
    #print(str_name0)
    fileout=[]
    yes_to_all_continue=""
    yes_to_all_error=""
    yes_to_all_error1=""
    for i in range(len(str_in_dir)):
        if not os.path.exists(f"{str_in_dir[i]}"):
            print("%*s"%(len_max,str_in_dir[i]),"No Calculation")
            fileout.append(["%*s"%(len_max,str_in_dir[i]),"No Calculation"])      
            continue
        job_on_hpc=os.popen("python %s/job_on_hpc.py %s"%(code_dir,str_in_dir[i])).readlines()[0].strip("\n")
        if (os.path.isfile("%s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i])))==1):
            etot1=(os.popen("grep JOB %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines())
            
            if len(etot1)!=1:
                etot0=(os.popen("grep \"! \" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i])))).readlines()
                read_error=0
                etot11=open("%s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()
                for error_line in  etot11:
                    #if error_line.find("BAD TERMINATION")!=-1:
                    #    read_error=2
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
                    ''' 
                    if (yes_to_all_error==""):
                        print("would you like me to continue cal for you?")
                        print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                        continue_error=input(">>>>>>")
                        if continue_error.lower()=="y":
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            #a=os.popen("pwd").readlines()
                            #print(a)
                            if mode=="relax":
                                os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                                print("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

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
                                os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                            
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
                            os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))

                        JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                        os.system("%s %s"%(sub_method,JOB_name))
                        os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                        os.chdir(dir)
                        print("%s subed!"%str_in_dir[i])
                    '''
                #output some data if there is no error
                elif read_error==0:
                    if len(etot0) > 0:
                        energy0=etot0[-1].split(" =")[-1].split("Ry")[0].strip()
                        energy_eV0=float(energy0)*27.211396/2
                        if job_on_hpc!="Not here":
                            print("%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    running",end='')
                        elif job_on_hpc=="Not here":
                            print("%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    disc quota exceeded",end='')
                        if len(os.popen("grep \"Gradient error\" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines())>0:
                            force_now=(os.popen("grep \"Gradient error\" %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()[-1]).split("Gradient error")[-1].strip().split("=")[-1].strip()
                            print("   ",force_now)
                        else:
                            print("force not generated!")
                        if job_on_hpc!="Not here":
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    running"])
                        elif job_on_hpc=="Not here":
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),energy0,"Ry","%.11f"%energy_eV0,"eV","    ERROR"])
                    elif len(etot0)==0:
                        if job_on_hpc!="Not here":
                            print("%*s"%(len_max,str_in_dir[i]),"running")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),"running"])
                        elif job_on_hpc=="Not here":
                            print("%*s"%(len_max,str_in_dir[i]),"ERROR")
                            fileout.append(["%*s"%(len_max,str_in_dir[i]),"ERROR"])
            #if there is JOB DONE output:
            else:
                if mode=="relax":
                    etot_relax_0=os.popen("grep ! %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()
                    if len(etot_relax_0)==0:#if no energy data output
                        print("!!!!!!!!!!",str_in_dir[i],"scf_not_converge")
                        etot="a = 0 Ry"
                    else:
                        etot=etot_relax_0[-1] 
                elif mode=="pdos":
                    print("%*s"%(len_max,str_in_dir[i]),"PDOS","    !!DONE!!!")
                    if wavefc_rm==1:
                        if os.path.exists("%s/%s.save"%(str_in_dir[i],prefix))==1:
                            os.system("rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            print("!!rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.mix*"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.mix*"%(str_in_dir[i],prefix))
                    fileout.append(["%*s"%(len_max,str_in_dir[i]),"PDOS","    !!DONE!!!"])
                    continue
                elif mode=="BaderCharge":
                    print("%*s"%(len_max,str_in_dir[i]),"BaderCharge","    !!DONE!!!")
                    if wavefc_rm==1:
                        if os.path.exists("%s/%s.save"%(str_in_dir[i],prefix))==1:
                            os.system("rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            print("!!rm -r %s/%s.save"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.mix*"%(str_in_dir[i],prefix))
                            os.system("rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.wfc*"%(str_in_dir[i],prefix))
                            print("!!rm %s/%s.mix*"%(str_in_dir[i],prefix))
                    fileout.append(["%*s"%(len_max,str_in_dir[i]),"BaderCharge","    !!DONE!!!"])
                    continue
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
                    etot0=(os.popen("grep ! %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines())
                    if len(etot0)==0:
                        print("%*s"%(len_max,str_in_dir[i]),"scf","    ERROR")
                        fileout.append(["%*s"%(len_max,str_in_dir[i]),"scf","    ERROR"])
                        continue
                    etot=etot0[0]
                    #etot=(os.popen("grep ! %s/%s/out_%s_%s"%(dir,(str_in_dir[i]),mode,(str_in_dir[i]))).readlines()[0])
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
                    '''
                    if (yes_to_all_continue==""):
                        print("would you like me to continue cal for you?")
                        print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                        continue_cal=input(">>>>>>")
                        if continue_cal.lower()=="y":
                            os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                            #a=os.popen("pwd").readlines()
                            #print(a)
                            if mode=="relax":
                                os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                                print("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
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
                                os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
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
                            os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                        JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                        os.system("%s %s"%(sub_method,JOB_name))
                        os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                        os.chdir(dir)
                        print("%s subed!"%str_in_dir[i])
                    '''
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
            '''
            if (yes_to_all_error1=="")&(error_waiting_recal==1):
                print("would you like me to restart cal for you?")
                print("plz input Y/y for yes, N/n for no, A/a for yes to all, Q/q for no to all")
                continue_error1=input(">>>>>>")
                if continue_error1.lower()=="y":
                    os.chdir("%s/%s/"%(dir,(str_in_dir[i])))
                    #a=os.popen("pwd").readlines()
                    #print(a)
                    if mode=="relax":
                        os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                        print("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
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
                        os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
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
                    os.system("python {code_dir}/bcon.py out_relax_%s in_relax_%s"%((str_in_dir[i]),(str_in_dir[i])))
                JOB_modify(JOB_name,mode,str_in_dir[i],type_hpc_out,ppn_tot_out)
                os.system("%s %s"%(sub_method,JOB_name))
                os.system("mv out_%s_%s out_buffer_%s"%(mode,str_in_dir[i],mode))
                os.chdir(dir)
                print("%s subed!"%str_in_dir[i])
            '''
    f_ou=open("pv_result_out","w+")
    for i in range(len(fileout)):
        for j in range(len(fileout[i])):
            f_ou.write(str(fileout[i][j])+"\t")
        f_ou.write("\n")
    print("pv_result_out generated!")
