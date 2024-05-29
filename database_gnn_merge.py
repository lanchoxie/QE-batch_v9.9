import os
import sys
import shutil
import pandas as pd
from tqdm import tqdm

# 检查 gnn_data.save 目录及其子目录
def check_gnn_data_save_dir():
    required_dirs = ['gnn_data.save/calculated_xsf_str_original',
                     'gnn_data.save/calculated_xsf_str_variance',
                     'gnn_data.save/xsf_str_original',
                     'gnn_data.save/xsf_str_variance']
    for dir in required_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Created directory: {dir}")

# 检查传入目录的有效性
def check_input_dirs(dirs):
    required_subdirs = ['calculated_xsf_str_original',
                        'calculated_xsf_str_variance',
                        'xsf_str_original',
                        'xsf_str_variance']
    for dir in dirs:
        for sub_dir in required_subdirs:
            if not os.path.exists(os.path.join(dir, sub_dir)):
                print(f"Error: Directory {dir} does not have the required subdirectory {sub_dir}.")
                sys.exit(1)
        if not os.path.isfile(os.path.join(dir, "Calculated_result_info.data")):
            print(f"Error: File Calculated_result_info.data does not exist in {dir}.")
            sys.exit(1)

# 复制文件
def copy_files(src_dirs):
    for src_dir in src_dirs:
        for subdir in ['calculated_xsf_str_original', 'calculated_xsf_str_variance', 'xsf_str_original', 'xsf_str_variance']:
            dest_dir = os.path.join('gnn_data.save', subdir)
            src_files = os.listdir(os.path.join(src_dir, subdir))
            for file in tqdm(src_files, desc=f"Copying files from {src_dir}/{subdir}"):
                shutil.copy(os.path.join(src_dir, subdir, file), dest_dir)

# 合并 Calculated_result_info.data 文件
def merge_data_files(src_dirs):
    data_frames = []
    for dir in src_dirs:
        data_frames.append(pd.read_csv(os.path.join(dir, "Calculated_result_info.data")))
    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data.to_csv('gnn_data.save/Calculated_result_info.data', index=False, encoding='utf-8-sig')
    print("gnn_data.save/Calculated_result_info.data created!")

if __name__ == "__main__":
    check_gnn_data_save_dir()
    input_dirs = sys.argv[1:]
    if not input_dirs:
        print("Error: No directories provided.")
        sys.exit(1)
    check_input_dirs(input_dirs)
    copy_files(input_dirs)
    merge_data_files(input_dirs)

