import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(sys.path[0])
from libs.modify_file import modify_parameter


#第一个参数是目标目录，第二个参数是out_relax_{str_name}.*的*对应的值，第三个参数是从后往前的值的个数
verbosity=1
plt_fig=0
critical_batch_value=10 #判断最近的几个relax收敛情况
spearman_crital_value=0.6 #小于这个值就判断为震荡

process=1

dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))

tot_dir=sys.argv[1].strip("/")

if len(sys.argv)<=2:
    file_tail=""
else:
    if sys.argv[2]=="0":
        file_tail=""
    else:
        file_tail="."+sys.argv[2]

if len(sys.argv)<=3:
    plt_fig_num=None
else:
    plt_fig_num=int(sys.argv[3])

str_name=tot_dir.split("/")[-1]

code_path=sys.path[0]

input_name=f"{tot_dir}/in_relax_{str_name}"
output_name=f"{tot_dir}/out_relax_{str_name}"

if os.path.isfile(f"{code_path}/bcon.py")==1:
    print("bcon.py found")
    find_bcon=1
if find_bcon==0:
    print("bcon.py not found,plz download the full package of QE-batch!")


dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))
type_hpc_out=os.popen(f"python {code_path}/create_job_script.py {str_name} relax").readline().strip("\n").strip().split("###")[-1]
if type_hpc_out=="slurm":
    sub_method="sbatch"
    del_method="scancel"
elif type_hpc_out=="pbs":
    sub_method="qsub"
    del_method="qdel"

replace_line=open("%s/replace.py"%code_path).readlines()
for i in replace_line:
    if i.find("JOB=")!=-1:
        JOB_name=i.split("JOB=")[-1].split("#")[0].split("\"")[1]
# 你的数据
#print(f"{tot_dir}/out_relax_{str_name}{file_tail}")
if not os.path.isfile(f"{tot_dir}/out_relax_{str_name}{file_tail}"):
    if not os.path.exists(f"{tot_dir}"):
        raise ValueError(f"No such dir as {tot_dir}")
    raise ValueError("No output generate yet!")
data_raw=os.popen(f"grep ! {tot_dir}/out_relax_{str_name}{file_tail}").readlines()
data=[float(i.split("=")[-1].split("Ry")[0].strip()) for i in data_raw]
if len(data)<=8:
    raise ValueError("No enough data!")

def cal_slope(crt_value):
    if len(data)<=crt_value:
        return "NaN",data,[],[],[]
    else:
        data_in=data[-1*crt_value:]
    # 准备数据
    x = np.arange(len(data_in)).reshape(-1, 1)
    y = np.array(data_in)
    
    # 创建并拟合线性回归模型
    model = LinearRegression()
    model.fit(x, y)
    # 获取斜率
    slope_in = model.coef_[0] 
    
        
    return slope_in,data_in,x,y,model
     
def calculate_correlation(input_list, correlation_type=None):
    n = len(input_list)
    reference_list = list(range(1, n + 1))  # 创建一个从1到n的列表

    if correlation_type == 'pearson':
        correlation_coefficient, _ = scipy.stats.pearsonr(input_list, reference_list)
    elif correlation_type == 'spearman':
        correlation_coefficient, _ = scipy.stats.spearmanr(input_list, reference_list)
    else:
        raise ValueError("Unsupported correlation type. Choose either 'pearson' or 'spearman'.")

    return correlation_coefficient

def judge(crt_value):
    slope,data_out,x_out,y_out,model_out=cal_slope(crt_value)
    corr_eff=calculate_correlation(data_out,correlation_type='spearman')
    #print("------------------------------------------")
    #print("Spearman coefficient:",corr_eff)
    if crt_value>len(data):
        crt_value=len(data)
    trend=""
    # 判断总体趋势
    if abs(corr_eff) < spearman_crital_value:
        trend="WARNING!! oscillating"
    elif abs(corr_eff) >= spearman_crital_value:
        if float(slope) < 0:
            trend="downward trend"
        elif float(slope) > 0:
            trend="WARNING!! upward trend"
        else:
            trend="Nan"
    else:
        trend="WARNING!! UNknow trend!!"
    if verbosity==1:
        print("****",f"Latest {crt_value} points,{trend}","SL:",f"{slope:.6}","SP",f"{corr_eff:.6}")
    if plt_fig_num==crt_value:
        #print(data_out)
        # 绘制数据和回归线
        #print()
        plt.figure(figsize=(12, 6))
        plt.title(f"{str_name}[{-1*plt_fig_num}:]: {trend}")
        plt.scatter(x_out, y_out, color='blue', label='Data points')
        plt.plot(x_out, y_out, color='blue', label=None)
        plt.plot(x_out, model_out.predict(x_out), color='red', label=f'Linear regression (slope: {slope:.2e})')
        plt.xlim([min(x_out)-(max(x_out)-min(x_out))*0.1,max(x_out)+(max(x_out)-min(x_out))*0.1])
        plt.ylim([min(y_out)-(max(y_out)-min(y_out))*0.1,max(y_out)+(max(y_out)-min(y_out))*0.1])
        plt.legend()
        plt.savefig("relax_E.png")
    return trend

judge_lst=[]
for i in range(4,20,2):
    judge_lst.append(judge(i))
warn_count=0
state=None
for j in judge_lst:
    if "WARNING" in j:
        warn_count+=1
    if warn_count>1:
        state="oscillation"

job_id=os.popen(f"python {code_path}/job_on_hpc_id.py {tot_dir}").readline().strip("\n").strip()
job_state=os.popen(f"python {code_path}/job_on_hpc.py {tot_dir}").readline().strip("\n").strip()
trust_rad=os.popen(f"grep trust_radius_max {input_name}").readlines()
trust_radius_max=0.8
for i in trust_rad:
    if "!" not in i:
        trust_radius_max=float(i.strip("\n").strip(",").split("=")[-1].strip())
if file_tail=="" and state=="oscillation":
    print(f"!!!Modify\n***********************{job_state}")
    if job_id!="Not here" and job_state!="waiting on line":
        print(f"{del_method} {job_id}")
        if process==1:
            os.system(f"{del_method} {job_id}")
        print(f"change into {tot_dir}")
        if process==1:
            os.chdir(tot_dir)
        print(f"modify in_relax_{str_name}, trust_radius_max={trust_radius_max*0.5:.3}")
        if process==1:
            modify_parameter("./",f"{str_name}","trust_radius_max",f"{trust_radius_max*0.5:.3}","&ions","    ")
        print(f"mv out_relax_{str_name} out_buffer")
        if process==1:
            os.system(f"python {code_path}/bcon.py out_relax_{str_name} in_relax_{str_name}")
            os.system(f"mv out_relax_{str_name} out_buffer")
        print(f"{sub_method} {JOB_name}")
        if process==1:
            os.system(f"{sub_method} {JOB_name}")
        print(f"change into {current_dir}")
        if process==1:
            os.chdir(current_dir)
        #modify_parameter(tot_dir,f"{str_name}","mixing_beta",0.2,"&electrons","    ")
    elif job_id=="Not here":
        print(f"change into {tot_dir}")
        if process==1:
            os.chdir(tot_dir)
        print(f"modify in_relax_{str_name}, trust_radius_max={trust_radius_max*0.5:.3}")
        if process==1:
            modify_parameter("./",f"{str_name}","trust_radius_max",f"{trust_radius_max*0.5:.3}","&ions","    ")
        print(f"mv out_relax_{str_name} out_buffer")
        if process==1:
            os.system(f"python {code_path}/bcon.py out_relax_{str_name} in_relax_{str_name}")
            os.system(f"mv out_relax_{str_name} out_buffer")
        #print(f"{sub_method} {JOB_name}")
        print(f"change into {current_dir}")
        if process==1:
            os.chdir(current_dir)
else:    
    print(f"!!!Nothing to be done\n***********************{job_state}")

if plt_fig_num!=None:
    os.system("python QE-batch/show_fig.py relax_E.png 12 6")
