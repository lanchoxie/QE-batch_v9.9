def modify_parameter(dir_, str_name, parameter, value, after_line, before_content):
    #***************************************#
    #输入的分别是：目标目录，目标文件，参数，参数修改值，参数（如果不存在）添加在哪个句子后面，参数前面需要添加什么
    #***************************************#

    input_file = f'{dir_}/in_relax_{str_name}'  # 输入文件名
    print("*********",input_file,dir_)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    parameter_line = None  # 初始化参数行为None
    parameter_found = False  # 标志参数是否已找到

    for i, line in enumerate(lines):
        if line.strip().startswith(after_line):  # 查找指定行
            parameter_line = i  # 记录参数应该插入或修改的位置
        if line.strip().startswith(parameter):  # 检查参数是否存在
            parameter_found = True
            parameter_line = i  # 更新参数行为当前行
            break  # 退出循环，因为已找到参数

    formatted_parameter = f'{before_content}{parameter} = {value}'  # 格式化参数

    if parameter_found:
        lines[parameter_line] = formatted_parameter + '\n'  # 修改存在的参数
    else:
        if parameter_line is not None:
            lines.insert(parameter_line + 1, formatted_parameter + '\n')  # 插入新参数
        else:
            lines.append(formatted_parameter + '\n')  # 或者在文件末尾添加新参数

    with open(input_file, 'w') as file:
        file.writelines(lines)  # 写入输出文件

def modify_flat(dir_, str_name, parameter, value, after_line, before_content):
    #***************************************#
    #输入的分别是：目标目录，目标文件，参数，参数修改值，参数（如果不存在）添加在哪个句子后面，参数前面需要添加什么
    #***************************************#

    input_file = f'{dir_}/{str_name}'  # 输入文件名
    #print("*********",input_file,dir_)
    with open(input_file, 'r') as file:
        lines = file.readlines()

    parameter_line = None  # 初始化参数行为None
    parameter_found = False  # 标志参数是否已找到

    for i, line in enumerate(lines):
        if line.strip().startswith(after_line):  # 查找指定行
            parameter_line = i  # 记录参数应该插入或修改的位置
        if line.strip().startswith(parameter):  # 检查参数是否存在
            parameter_found = True
            parameter_line = i  # 更新参数行为当前行
            break  # 退出循环，因为已找到参数

    formatted_parameter = f'{before_content}{parameter}={value}'  # 格式化参数

    if parameter_found:
        lines[parameter_line] = formatted_parameter + '\n'  # 修改存在的参数
    else:
        if parameter_line is not None:
            lines.insert(parameter_line + 1, formatted_parameter + '\n')  # 插入新参数
        else:
            lines.append(formatted_parameter + '\n')  # 或者在文件末尾添加新参数

    with open(input_file, 'w') as file:
        file.writelines(lines)  # 写入输出文件

if __name__ == "__main__":
    modify_parameter("dirs", 'example', 'trust_radius_max', 8, '&ions')

