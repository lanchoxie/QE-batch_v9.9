import os
import sys
import pymatgen
import numpy as np
from collections import Counter



extract_energy=1
extract_xsf_file=1
creating_gnn_graph=0

db_name='gnn_data.save/wholegraphs_53d_features.db'
code_dir=sys.path[0]
dir_file=(os.popen("pwd").read())
current_dir=max(dir_file.split('\n'))

gnn_db='gnn_data.save'
if not os.path.exists(gnn_db):
    os.system(f"mkdir {gnn_db}")

original_xsf_dir=os.path.join(gnn_db,"calculated_xsf_str_original")
variance_xsf_dir=os.path.join(gnn_db,"calculated_xsf_str_variance")
original_xsf_dir_before=os.path.join(gnn_db,"xsf_str_original")
variance_xsf_dir_before=os.path.join(gnn_db,"xsf_str_variance")

if not os.path.exists(original_xsf_dir):
    os.system(f"mkdir {original_xsf_dir}")
if not os.path.exists(variance_xsf_dir):
    os.system(f"mkdir {variance_xsf_dir}")
if not os.path.exists(original_xsf_dir_before):
    os.system(f"mkdir {original_xsf_dir_before}")
if not os.path.exists(variance_xsf_dir_before):
    os.system(f"mkdir {variance_xsf_dir_before}")


def read_files(infiles,split_syb,split_mode=None):
    f=open(infiles,"r+")
    f1=f.readlines()
    read_data=[]
    for lines in f1:
        read_data_row=[]
        if "Direct" in lines:
            continue
        if "***" in lines:
            continue
        if len(lines) <= 1:
            continue
        if "\t" in lines:
            a_bf=(lines.split("\n")[0]).split("\t")
        else:
            a_bf=(lines.split("\n")[0]).split(split_syb)
        for i in a_bf:
            if len(i)>0:
                if split_mode=="str" or not split_mode:
                    read_data_row.append(str(i.strip()))
                elif split_mode=="float":
                    read_data_row.append(float(i.strip()))
        read_data.append(read_data_row)
    return read_data

def parse_variants_name(name):
    parts = name.split('_')
    info = {}
    for part in parts[1:]:
        if 'Li' in part or 'Ni' in part:
            element, num = part[:2], part[2:]
            info[element] = int(num)
    return info


if extract_energy==1:
    str_states_raw=read_files("pv_result_out",' ',split_mode='str')
    str_energy={}
    for i in str_states_raw:
        if i[-1].find('DONE')!=-1:
            str_energy[i[0]]={'energy':float(i[-3]),'variance': []}
    
    for key, value in str_energy.items():
        folder_path = os.path.join('exchange_dir', f"{key}-exhange_file-extractor")
        if not os.path.isdir(folder_path):
            print(f"{folder_path} not found, tring exhange_file")
            folder_path = os.path.join('exchange_dir', f"{key}-exhange_file")
        if os.path.isdir(folder_path):
            variant_file = os.path.join(folder_path, "pv_result_out")
            if not os.path.exists(variant_file):
                print(f"{variant_file} not found,trying using read_relax_E_for_UI.py...")
                os.chdir(folder_path)
                jj=os.system("python QE-batch/read_relax_E_for_UI.py")
                os.chdir(current_dir)
            if os.path.exists(variant_file):
                variants_raw = read_files(variant_file, ' ', split_mode='str')
                for variant in variants_raw:
                    if 'DONE' in variant[-1]:
                        variant_info = parse_variants_name(variant[0])
                        variant_energy = float(variant[-3])
                        energy_diff = variant_energy - value['energy']
                        variant_info['energy'] = variant_energy
                        variant_info['energy_diff'] = energy_diff
                        value['variance'].append(variant_info)
    
        elif not os.path.isdir(folder_path):
            print(f"{folder_path} not found!!")
    
    count=0
    for i,v in str_energy.items():
        print(i,len(v['variance']))
        count+=len(v['variance'])
    print("Total data point:",count)
    
    calculated_data=[]
    calculated_data.append(["str_name","Li","Ni","delta_E","original_E","variance_E"])
    for i,v in str_energy.items():
        for j,w in enumerate(v['variance']):
            calculated_data.append([i,v['variance'][j]['Li'],v['variance'][j]['Ni'],v['variance'][j]['energy_diff'],v['energy'],v['variance'][j]['energy']])
    f1=open(f"{gnn_db}/Calculated_result_info.data","w+")
    for i in calculated_data:
        f1.writelines("\t".join([str(x) for x in i])+"\n")


def vasp2xsf(str_name):
    def split_lines(infiles,split_syb=None,split_mode=None):
        #f=open(infiles,"r+")
        #f1=f.readlines()
        if split_syb==None:
            split_syb=' '
        f1=infiles
        read_data=[]
        for lines in f1:
            read_data_row=[]
            if "Direct" in lines:
                continue
            if "***" in lines:
                continue
            if len(lines) <= 1:
                continue
            if "\t" in lines:
                #print(lines,"t")
                a_bf=(lines.split("\n")[0]).split("\t")
            else:
                #print(lines,"None")
                a_bf=(lines.split("\n")[0]).split(split_syb)
            for i in a_bf:
                if len(i)>0:
                    if split_mode=="str" or not split_mode:
                        read_data_row.append(str(i.strip()))
                    elif split_mode=="float":
                        read_data_row.append(float(i.strip()))
                    elif split_mode=="int":
                        read_data_row.append(int(i.strip()))
            read_data.append(read_data_row)
        return read_data
    
    raw_data=open(str_name).readlines()
    lattice=split_lines(raw_data[2:5],split_mode='float')
    element=split_lines([raw_data[5]],split_mode='str')
    element_count=split_lines([raw_data[6]],split_mode='int')
    file_style=raw_data[7].strip("\n").strip().lower()
    #file_style_list=split_lines([raw_data[7]],split_mode='str')
    #file_style=file_style_list[0][0].lower()
    coords=split_lines(raw_data[8:],split_mode='float')
    element_list=[]
    for i,v in enumerate(element[0]):
        element_list.extend([v]*element_count[0][i])
    
    if file_style=='direct':
        lattice_matrix=np.array(lattice)
        coords_car=[]
        # 转换每个分数坐标
        for coord in coords:
            # 将分数坐标转换为 Numpy 数组
            frac_coords_array = np.array(coord[:3])
            # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
            cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
            # 将笛卡尔坐标添加到结果列表
            coords_car.append(cart_coords_array.tolist())
    elif file_style=='cartesian':
        coords_car=coords.copy()
    else:
        raise ValueError("Unknow formation!")
    
    output=[]
    output.append("DIM-GROUP\n")
    output.append("           3           1\n")
    output.append(" PRIMVEC\n")
    for i in lattice:
        output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
    output.append(" PRIMCOORD\n")
    output.append(f"          {len(coords)}           1\n")
    for i,v in enumerate(coords_car):
        output.append(f" {element_list[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 
    outname=str_name.replace(".vasp",".xsf")
    f1=open(outname,"w+")
    for i in output:
        f1.writelines(i)    
    return outname

if extract_xsf_file==1:
    print("Extracting xsf files from output file...")
    for ori_file,v in str_energy.items():
        #print("\n"+ori_file)
        outname=vasp2xsf(ori_file+'.vasp') 
        var_path=os.path.join("exchange_dir",f"{ori_file}-exhange_file-extractor") 
        if not os.path.isdir(var_path):
            var_path = os.path.join('exchange_dir', f"{ori_file}-exhange_file")

        os.system(f"mv {outname} {original_xsf_dir_before}")
        if os.path.isfile(f"{ori_file}/out_relax_{ori_file}.xsf"):
            os.system(f"cp {ori_file}/out_relax_{ori_file}.xsf {original_xsf_dir}") 
        else:
            jj=os.popen(f"python {code_dir}/qe_out_reader_gnn.py {ori_file} xsf").readlines()
            os.system(f"cp out_relax_{ori_file}.xsf {ori_file}") 
            os.system(f"mv out_relax_{ori_file}.xsf {original_xsf_dir}") 
        for var_file,cont in enumerate(v['variance']):
            print(f"{ori_file}: {var_file+1}/{len(v['variance'])}",'\r',end='')
            var_file_name=f"{ori_file}_Li"+str(v['variance'][var_file]['Li'])+"_Ni"+str(v['variance'][var_file]['Ni'])
            #print(var_file_name)
    
            outname_i=vasp2xsf(f"{var_path}/{var_file_name}.vasp")
            os.system(f"mv {outname_i} {variance_xsf_dir_before}") 
            if os.path.isfile(f"{var_path}/{var_file_name}/our_relax_{var_file_name}.xsf"):
                os.system(f"cp {var_path}/{var_file_name}/our_relax_{var_file_name}.xsf {variance_xsf_dir}") 
            else:
                jj=os.popen(f"python {code_dir}/qe_out_reader_gnn.py {var_path}/{var_file_name} xsf").readlines()
                os.system(f"cp out_relax_{var_file_name}.xsf {var_path}/{var_file_name}") 
                os.system(f"mv out_relax_{var_file_name}.xsf {variance_xsf_dir}") 
        print()


def decode_lattice_and_coords(vector):
    """
    从12维向量中解码晶格矩阵和坐标。
    """
    lattice_matrix = np.array(vector[-12:-3]).reshape((3, 3))
    coords = np.array(vector[-3:])
    return lattice_matrix, coords

def calculate_minimum_distance(vector1, vector2):
    """
    计算考虑周期性边界条件的两个原子之间的最短距离。
    """
    lattice_matrix, coords1 = decode_lattice_and_coords(vector1)
    _, coords2 = decode_lattice_and_coords(vector2)
    
    min_distance = np.inf
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # 计算相邻晶胞中原子B的坐标
                translated_coords = coords2 + i * lattice_matrix[0, :] + j * lattice_matrix[1, :] + k * lattice_matrix[2, :]
                # 计算与固定原子A的距离
                distance = np.linalg.norm(coords1 - translated_coords)
                # 更新最小距离
                if distance < min_distance:
                    min_distance = distance
    return min_distance

#calculated_data was used in this code paragraph
if creating_gnn_graph==1:
    print("Creating GNN graph...")

    energy_data=read_files('gnn_data.save/Calculated_result_info.data',split_syb='\t',split_mode='str')

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from time import sleep
    import random  # 假设的函数可能会用到random，只是示例
    from pymatgen.core.periodic_table import Element
    from mendeleev import element as MEN_element
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.core.structure import Structure
    from pymatgen.io.xcrysden import XSF
    import pandas as pd
    import networkx as nx
    import numpy as np
    import torch
    import torch.nn as nn
    import plotly.graph_objs as go
    import sqlite3
    import json
    import math
    import copy
    import os
    import warnings
    
    # 忽略所有UserWarning警告
    warnings.filterwarnings("ignore", category=UserWarning, module='pymatgen.*')
    
    # 忽略所有DeprecationWarning警告
    warnings.filterwarnings("ignore", category=DeprecationWarning, module='pymatgen.*')
    
    # 运行你的代码

    def create_whole_graphs(str_name_i):
          
        verbosity=1
        
        str_name=str_name_i
        str_index=[i[0] for i in energy_data].index(str_name)
    
        xsf_file_orig = f'gnn_data.save/xsf_str_original/{str_name}.xsf'
        xsf_file_relax = f'gnn_data.save/calculated_xsf_str_original/out_relax_{str_name}.xsf'
        # 加载 .data 文件
        # 储存li_ni_edge能量信息
        if not os.path.isfile("gnn_data.save/Calculated_result_info.data"):
            raise ValueError("No such file as gnn_data.save/Calculated_result_info.data!")
    
        exchange_data_raw = pd.read_csv("gnn_data.save/Calculated_result_info.data",sep='\t')    
        exchange_data = exchange_data_raw[exchange_data_raw['str_name'] == str_name]
        #加载DFT .data文件读入bader和mag
        #input_features={}  
        #dir_charge=dir_+"charges_mags\\"
        #file=f"bader_charge_of_{str_name}.data"
        #bader_data = pd.read_csv(dir_charge+file,sep='\s+',header=None,engine='python')
        #input_features['atom_index']=bader_data.iloc[:,0].apply(lambda x: int(x.split('-')[1]) - 1)   
        #input_features['atom_ele']=bader_data.iloc[:,0].apply(lambda x: x.split('-')[0])  
        #input_features['bader_valance']=bader_data.iloc[:,3]
        #input_features['lowdin_valance']=bader_data.iloc[:,4]    
        #input_features['magnetic']=bader_data.iloc[:,5] 


        if verbosity==1:
            print(str_name,": Normal")
        #print(exchange_data.columns)
    
        def read_xsf(file_path):
            with open(file_path, 'r') as file:
                xsf_content = file.read()
            xsf = XSF.from_str(xsf_content)
            return xsf.structure
    
        crystal_original = read_xsf(xsf_file_orig)
        crystal_relax = read_xsf(xsf_file_relax)
        # Create a deep copy of the crystal structure
        crystal = copy.deepcopy(crystal_original)
    
        def create_original_index_list(crystal):
            original_indices = []
            for i, site in enumerate(crystal):
                if str(site.specie) != 'O':  # 只为非氧原子创建映射
                    original_indices.append(i)
            return original_indices
    
        #index projection from original to delete
        original_index_map = create_original_index_list(crystal)
    
        def map_old_indices_to_new(original_index_map, crystal):
            new_index_map = {}
            for new_index, old_index in enumerate(original_index_map):
                new_index_map[old_index] = new_index
            return new_index_map
    
        #index projection from Oxygen_delete structure to original structure
        new_index_map = map_old_indices_to_new(original_index_map, crystal)
    
        # Remove all oxygen sites from the copy
        crystal_metal = copy.deepcopy(crystal_original)
        crystal_metal.remove_species(["O"])
    
        # 使用pymatgen的CrystalNN找到临近的原子
        crystal_nn = CrystalNN()
        neighbors = [crystal_nn.get_nn_info(crystal, n=i) for i, _ in enumerate(crystal.sites)]
    
        neighbors_metal = [crystal_nn.get_nn_info(crystal_metal, n=i) for i, _ in enumerate(crystal_metal.sites)]
    
    
        # 构建图
        G = nx.Graph()
        for i, site in enumerate(crystal_original.sites):
            #lattice_info = crystal_original.lattice.matrix
            lattice_info = crystal_original.lattice.matrix.flatten()  # 将3x3矩阵拉平成9维列表
            coord = site.coords  # 获取原子的坐标，本身是一个三维向量
    
            #lattice_info_orig = crystal_original.lattice.matrix
            lattice_info_relax = crystal_relax.lattice.matrix.flatten()  # 将3x3矩阵拉平成9维列表
            coord_relax = crystal_relax.sites[i].coords  # 获取原子的坐标，本身是一个三维向量
            lattice_coord = list(lattice_info) + list(coord) + list(lattice_info_relax) + list(coord_relax) + [i]  # 将两者合并成12维列表+1
            G.add_node(i, element=site.specie.symbol, lattice_coord=lattice_coord)
            #G.nodes[node]['lattice_coord'] = lattice_coord
            
            for neighbor in neighbors[i]:
                neighbor_index = neighbor['site_index']
                if not G.has_edge(i, neighbor_index):
                    G.add_edge(i, neighbor_index, edge_type='original')
    
        # 构建特定原子互换能的边
        for i, site in enumerate(crystal_metal.sites):
            for neighbor in neighbors_metal[i]:
                neighbor_index = neighbor['site_index']
                if ((site.specie == Element("Li")) and (crystal_metal[neighbor_index].specie == Element("Ni"))) or \
                   ((site.specie == Element("Ni")) and (crystal_metal[neighbor_index].specie == Element("Li"))):
                    if not G.has_edge(original_index_map[i], original_index_map[neighbor_index]):
                        G.add_edge(original_index_map[i], original_index_map[neighbor_index], edge_type='li_ni_edge')
                        
        #print("111")
    
        # 初始化Ni，Mn，Ti，Co的计数为0
        counts = {"Li":0, "O":0, "Mg":0, "Al":0, "Ti":0, "V":0, "Mn":0, "Co":0, "Ni":0, "Zr":0}
        
        elements_total=[str(i.specie) for i in crystal_original.sites]    
        # 使用Counter来统计列表中每个元素的出现次数
        element_counter = Counter(elements_total)
        # 更新counts字典中的计数
        counts.update(element_counter)
        # 打印结果
        #print({element: count for element, count in counts.items()})      
        concen=[count for element, count in counts.items()]
        #concen.append(str_name_i)    
        #print(concen)
        
        #节点信息嵌入：元素独热编码+电负性
    
        # 预设的元素列表
        elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
        # 创建独热编码字典
        element_to_onehot = {elem: [int(i == elem) for i in elements_list] for elem in elements_list}
    
        # 为图中的每个节点添加特征
        for node in G.nodes():
            element = G.nodes[node]['element']  # 获取节点的元素类型
            one_hot = element_to_onehot[element]  # 获取独热编码
            #if input_features['atom_ele'][node]!=element or input_features['atom_index'][node]!=node:
                #if verbosity==1:        
                    #print(str_name,f": Mismatch of element:{element},index:{node},pass")
                    #return None
            
            #嵌入DFT的数据：bader价态，lowdin价态和磁矩
            #bader_valance_i=[input_features['bader_valance'][node]]
            #lowdin_valance_i=[input_features['lowdin_valance'][node]]
            #magnetic_i=[input_features['magnetic'][node]]
            
            #print(bader_valance_i,node,element)
            
            # 获取其他化学属性
            pymatgen_elem = Element(element)
            mendeleev_elem = MEN_element(element)
    
            electronegativity = [pymatgen_elem.X if pymatgen_elem.X else 0]  # 电负性
            atomic_radius = [pymatgen_elem.atomic_radius if pymatgen_elem.atomic_radius else 0]  # 原子半径
            ionization_energy = [mendeleev_elem.ionenergies.get(1, 0)]  # 离子化能
            atomic_mass = [mendeleev_elem.atomic_weight if mendeleev_elem.atomic_weight else 0]  # 原子质量
            melting_point = [mendeleev_elem.melting_point if mendeleev_elem.melting_point else 0]  # 熔点
            density = [mendeleev_elem.density if mendeleev_elem.density else 0]  # 密度
            thermal_conductivity = [mendeleev_elem.thermal_conductivity if mendeleev_elem.thermal_conductivity else 0]  # 热导率
    
            # 合并特征
            lattice_coord = G.nodes[node]['lattice_coord'] # 获取原子坐标和晶格信息
            features = one_hot + concen + electronegativity + atomic_radius + ionization_energy + atomic_mass + melting_point + density + thermal_conductivity + lattice_coord + [str_name]
            #features = one_hot + concen + electronegativity + atomic_radius + ionization_energy + atomic_mass + melting_point + density + thermal_conductivity + bader_valance_i + lowdin_valance_i + magnetic_i +  lattice_coord + str_name
            #features = one_hot + concen + electronegativity + atomic_radius + ionization_energy + atomic_mass + melting_point + density + thermal_conductivity + lattice_coord
            
            G.nodes[node]['feature'] = features
    
            # 特征向量的维度说明：
            # - 独热编码：10维（对应10种元素）
            # - 浓度：10维
            
            # - 电负性：1维
            # - 原子半径：1维
            # - 离子化能：1维
            # - 原子质量：1维
            # - 熔点：1维
            # - 密度：1维
            # - 热导率：1维
 
            #####not used down###
            ##### - bader价态：1维
            ##### - lowdin价态：1维
            ##### - 磁矩：1维
            #####not used up####

            # - 晶格-优化前：9维
            # - 坐标-优化前：3维
            # - 晶格-优化后：9维
            # - 坐标-优化后：3维
            # - 原子序号：1维
 
            # - 结构名称：1维
            # 总共：53维
        
    
        # 调整原子索引为从 0 开始（减去 1）
        exchange_data_copy=exchange_data.copy()
        exchange_data_copy.loc[:, 'atom_1_index'] = exchange_data['Li']-1
        exchange_data_copy.loc[:, 'atom_2_index'] = exchange_data['Ni']-1
    
        # 提取原子对和互换能
        exchange_pairs = exchange_data_copy[['atom_1_index', 'atom_2_index', 'delta_E']]
    
        #边信息嵌入：

        #将距离添加到edge中
        #for u, v, data in G.edges(data=True):
            #distance = calculate_minimum_distance(G.nodes[u]['lattice_coord'], G.nodes[v]['lattice_coord'])
            #G[u][v]['distance'] = distance 
            
        # 将互换能信息添加到li_ni_edge边中
        for _, row in exchange_pairs.iterrows():
            atom_1_index = row['atom_1_index']
            atom_2_index = row['atom_2_index']
            exchange_energy = row['delta_E']
    
            
            # 检查边是否存在并且是li_ni_edge类型的边
            if G.has_edge(atom_1_index, atom_2_index) and G[atom_1_index][atom_2_index].get('edge_type') == 'li_ni_edge':
                #print(G.nodes[atom_1_index]['element']+str(int(atom_1_index+1)),G.nodes[atom_2_index]['element']+str(int(atom_2_index+1)))
                G[atom_1_index][atom_2_index]['delta_E'] = exchange_energy    
                G[atom_1_index][atom_2_index]['str_name'] = str_name
                
                
        #print(G.nodes[0]['feature'])
        #edges = list(G.edges(data=True))
        #print(edges)
        # 获取第一条边及其属性
        #first_edge = edges[0]
    
        # 解包第一条边的信息
        #node1, node2, attr_dict = first_edge
    
        # 获取 distance 属性
        #distance1 = attr_dict['distance']
        #print(node1,node2,distance1)
        
        return G

    def store_wholegraphs_in_db(wholegraphs):
        # 连接到SQLite数据库
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
    
        # 创建表格
        c.execute('''CREATE TABLE IF NOT EXISTS wholegraphs
                     (id INTEGER PRIMARY KEY, name TEXT, graph_data TEXT)''')
    
        # 清空表中的现有记录
        c.execute("DELETE FROM wholegraphs")
    
        # 存储每个子图
        for wholegraph in wholegraphs:
            name, graph = wholegraph[0], wholegraph[1]
            # 将图数据转换为JSON格式
            graph_data = json.dumps(nx.node_link_data(graph))
            # 插入图的名称和图数据
            c.execute("INSERT INTO wholegraphs (name, graph_data) VALUES (?, ?)", (name, graph_data))
    
        # 提交事务并关闭连接
        conn.commit()
        conn.close()

    def process_graph(str_name_i):
        graph=create_whole_graphs(str_name_i)
        return graph
    
    def print_progress(done, total):
        percent_done = done / total * 100
        bar_length = int(percent_done / 100 * 60)
        bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
        print(bar, "\r", end='')
    
    def convert_graphs(graphs):
        with ProcessPoolExecutor() as executor:
            # 创建future到索引的映射
            futures = {executor.submit(process_graph, graph): i for i, graph in enumerate(graphs)}
            converted_graphs = [None] * len(graphs)  # 预先分配结果列表
            total_done = 0  # 已完成的任务数量
            
            while total_done < len(futures):
                done_futures = [f for f in futures if f.done()]  # 获取所有已完成的futures
                for future in done_futures:
                    index = futures[future]  # 获取原始图的索引
                    if converted_graphs[index] is None:  # 检查是否已更新进度
                        result = future.result()
                        converted_graphs[index] = result if result is not None else None
                        total_done += 1
                        print_progress(total_done, len(graphs))  # 打印进度
                sleep(0.1)  # 稍微等待以减少CPU使用率
    
        return [[graphs[i],graph] for i,graph in enumerate(converted_graphs) if graph is not None]
    
    # 使用函数处理这些图
    print("Creating whole graphs...")
    converted_graphs = convert_graphs(list(set([str_info[0] for str_info in energy_data[1:]]))) 
    print(f"\nStoring whole graphs in {db_name}...")
    store_wholegraphs_in_db(converted_graphs) 
