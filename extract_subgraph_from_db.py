import argparse
import json
import sqlite3
import numpy as np
import networkx as nx
import os
import sys
from collections import Counter

parser = argparse.ArgumentParser(description="Swap Li-Ni pairs in crystal structures.")
parser.add_argument("graph_ind",type=int, help="The name of the graph structure to retrieve.")
parser.add_argument("out_type", choices=["xsf", "xsf_std", "vasp"], help="The format of the output structure.")
parser.add_argument("knn", nargs='?', default=2, type=int,choices=[2,3,4,5], help="The K-th nearest neighbor,default 2.")
parser.add_argument("--perfect", "-p", action="store_true", help="Output the perfect (pre-optimization) structure.")
parser.add_argument("--fig", "-f", action="store_true", help="Output the subgraph figures.")

args = parser.parse_args()

#png settings
#resolution=[12,8]
resolution=[14,7]
png_font=12

k_nn=args.knn
db_path = f'gnn_data.save/subgraphs_k_neighbor_{k_nn}_gnn_53d_feature_train.db'
output_dir='gnn_data.save/extract_subgraph_str_buffer'
output_fig_dir='gnn_data.save/extract_subgraph_figure_buffer'
if not os.path.exists(output_dir):
    os.system(f"mkdir {output_dir}")
if not os.path.exists(output_fig_dir):
    os.system(f"mkdir {output_fig_dir}")

output_type=args.out_type
ind=args.graph_ind

create_figs=args.fig

def load_data_from_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT graph_data FROM subgraphs")
    all_graphs = c.fetchall()
    conn.close()
    return all_graphs

def create_networkx_graphs(all_graphs):
    subgraphs = []
    for graph_json_tuple in all_graphs:
        graph_json_str = graph_json_tuple[0]  # 从元组中提取 JSON 字符串
        graph_data = json.loads(graph_json_str)  # 将 JSON 字符串解析为 Python 对象
        subgraph = nx.node_link_graph(graph_data)  # 使用 networkx 创建图
        subgraphs.append(subgraph)
    
    return subgraphs

all_graphs  = load_data_from_db(db_path)
converted_subgraphs = create_networkx_graphs(all_graphs)
print("Total exchange_E applied subgraph number:",len(converted_subgraphs))
graph_name=converted_subgraphs[ind].nodes[0]['feature'][-1]
node_number=len(converted_subgraphs[ind].nodes)
#print(graph_name)
#print([[u,v,data] for u,v,data in converted_subgraphs[ind].edges(data=True)])
#print(converted_subgraphs[ind].nodes[102]['feature'])
#sys.exit(0)

def rl_aw(element_i):
    element=["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Uun","Uuu","Uub"]   #end in 112
    #return element[int(element_i-1)]
    return element.index(element_i)+1


def extract_lattice_and_coords(graph):
    # 晶格信息：'lattice_optimized'对应优化后的晶格矩阵，每个元素是一个包含3个值的列表（9维特征）
    # 节点信息：每个节点有一个'coordinates_optimized'特征，包含优化后的坐标（3维特征）
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    
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
    
    # 初始化列表来存储晶格和坐标信息
    perf_lat = []
    opt_lat = []
    perf_coords = []
    opt_coords = []
    perf_element = []
    opt_element = []
    index=[]
    for node, data in graph.nodes(data=True):
        features = data['feature']

        # 提取原子种类
        element_index = np.argmax(features[:10])  # 假设前10维是独热编码
        element = elements_list[element_index]  # 获取元素种类

        # 添加元素信息
        perf_element.append(element)
        opt_element.append(element)

        # 提取晶格信息
        if not perf_lat:  # 假设所有节点的晶格信息相同，只需提取一次
            perf_lat.extend(features[-26:-17])  # 提取优化前晶格信息
            opt_lat.extend(features[-14:-5])  # 提取优化后晶格信息
        
        # 提取坐标信息
        perf_coords.append(features[-17:-14])  # 提取优化前坐标信息
        opt_coords.append(features[-5:-2])  # 提取优化后坐标信息
        # 提取原子索引
        index.append(features[-2])

    # 打包，然后根据 index 排序
    packed_list = list(zip(index, perf_coords, opt_coords, perf_element, opt_element))
    packed_list.sort(key=lambda x: x[0])  # 假设index是可排序的
    
    # 解包
    index, perf_coords, opt_coords, perf_element, opt_element = zip(*packed_list)
    
    # 转换回所需的格式
    perf_coords = list(perf_coords)
    opt_coords = list(opt_coords)
    perf_element = list(perf_element)
    opt_element = list(opt_element)
    index = list(index)  # 如果你还需要使用排序后的索引列表

    # 将晶格信息转换为3x3的numpy数组
    perf_lat = np.array(perf_lat).reshape((3, 3))
    opt_lat = np.array(opt_lat).reshape((3, 3))

    # 组装返回的字典
    result = {
        'perf_lat': perf_lat,
        'opt_lat': opt_lat,
        'perf_coords': perf_coords,
        'opt_coords': opt_coords,
        'perf_element': perf_element,
        'opt_element': opt_element
    }

    return result


def frac_to_car(lat,coords):
    lattice_matrix=lat
    coords_car=[]
    # 转换每个分数坐标
    for coord in coords:
        # 将分数坐标转换为 Numpy 数组
        frac_coords_array = np.array(coord[:3])
        # 通过晶格矩阵将分数坐标转换为笛卡尔坐标
        cart_coords_array = np.dot(frac_coords_array, lattice_matrix)
        # 将笛卡尔坐标添加到结果列表
        coords_car.append(cart_coords_array.tolist())
    return coords_car

def car_to_frac(lat,coords):
    lattice_matrix=lat
    coords_frac=[]
    for coord in coords:
        car_coords_array = np.array(coord[:3])
        # 计算晶格矩阵的逆
        lattice_matrix_inv = np.linalg.inv(lattice_matrix)
        # 使用晶格矩阵的逆将笛卡尔坐标转换回分数坐标
        frac_coords_array = np.dot(car_coords_array,lattice_matrix_inv)
        coords_frac.append(frac_coords_array.tolist())
    return coords_frac

def find_center_atoms(subgraph):
    # 假设中心原子是连接li_ni_edge的两个原子
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            return u, v, data
    return None, None

def shift_cluster_to_box_center(subgraph, perfect=False):  #Not woring: I want to shift the separate cluster into the middle of the box
      
    features=subgraph.nodes[0]['feature']
    lat_list=[]
    if not lat_list:  # 假设所有节点的晶格信息相同，只需提取一次
        lat_list.extend(features[-26:-17] if perfect else features[-14:-5])  # 提取晶格信息
    lat=np.array(lat_list).reshape((3,3))
    #perf_coords.append(features[-17:-14] if perfect else features[-5:-2])  # 提取坐标信息
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    
    # Step 1: Find the central atoms
    u, v, data_subgraph_lini_edge = find_center_atoms(subgraph)
    if u is None or v is None:
        return []  # No central atoms found
 
    # Step 2: Retrieve Cartesian coordinates of central atoms
    u_coords = subgraph.nodes[u]['feature'][-17:-14] if perfect else subgraph.nodes[u]['feature'][-5:-2]
    v_coords = subgraph.nodes[v]['feature'][-17:-14] if perfect else subgraph.nodes[v]['feature'][-5:-2]
    u_element_index = np.argmax(subgraph.nodes[u]['feature'][:10])  # 假设前10维是独热编码
    u_element = elements_list[u_element_index]  # 获取元素种类
    v_element_index = np.argmax(subgraph.nodes[v]['feature'][:10])  # 假设前10维是独热编码
    v_element = elements_list[v_element_index]  # 获取元素种类
 
    # Step 3: Calculate the midpoint (center of the cluster)
    center = [(u + v) / 2 for u, v in zip(u_coords, v_coords)]
    print("*"*60)
    print(f"Extract from structure:{graph_name}")
    print(f"K-th neighbor:{k_nn}\nNode_number in subgraph:{node_number}")
    print(u+1,u_element,u_coords)
    print(v+1,v_element,v_coords)
    print(data_subgraph_lini_edge['delta_E'])
    print("*"*60)
    #shift in fraction condition
 
    # Convert cluster center to fractional coordinates
    #center_frac = car_to_frac(lat, [center])[0]
 
    # Step 5: Calculate the shift needed (target is center of the lattice in fractional coordinates)
    #shift = [0.5 - x for x in center_frac]

    #shift in catesian condition
    # 计算晶格中心在笛卡尔坐标系下的位置
    lattice_center = np.dot([0.5, 0.5, 0.5], lat)
    #print(lattice_center)
    #print(center)
    # 计算集群中心到晶格中心的位移（在笛卡尔坐标系下）
    shift = lattice_center - center
    #print(shift)
    # Step 6: Apply this shift to all atoms

    # Step 1: Collect all coordinates
    #all_coords = [subgraph.nodes[node]['feature'][-17:-14] if perfect else subgraph.nodes[node]['feature'][-5:-2] for node in subgraph.nodes]
    
    # Step 2: Find the maximum negative (i.e., the most negative or smallest) value in each dimension
    #max_negative = [min(coord[i] for coord in all_coords if coord[i] < 0) for i in range(3)]
    
    # Step 3: Adjust coordinates by subtracting the maximum negative value if it exists; otherwise, do not adjust
    #adjusted_coords = []
    #for coord in all_coords:
    #    adjusted_coord = [(coord[i] - max_negative[i]) if max_negative[i] < 0 else coord[i] for i in range(3)]
    #    adjusted_coords.append(adjusted_coord)
    
    # Assuming 'shift' is calculated elsewhere and is the vector to center the cluster
    # Step 4: Apply shift to the adjusted coordinates
    #shifted_coords = [[coord[i] + shift[i] for i in range(3)] for coord in adjusted_coords]
    #return shifted_coords
    shifted_coords = []
    for node in subgraph.nodes:
        node_coords = subgraph.nodes[node]['feature'][-17:-14] if perfect else subgraph.nodes[node]['feature'][-5:-2]
        #node_coords_frac = car_to_frac(lat, [node_coords])[0]  # Convert to fractional
        #shifted_frac = [coord + shift_val for coord, shift_val in zip(node_coords_frac, shift)]  # Apply shift
        #shifted_coords_car = frac_to_car(lat, [shifted_frac])  # Convert back to Cartesian
        #shifted_coords.append(shifted_coords_car[0])
        shifted_coord = [coord + shift_val for coord, shift_val in zip(node_coords, shift)]  # Apply shift
        shifted_coords.append(shifted_coord)
    return shifted_coords

def adjust_coords_to_positive(subgraph, perfect):  #Not woring either
    # Step 1: Collect all coordinates
    all_coords = [subgraph.nodes[node]['feature'][-17:-14] if perfect else subgraph.nodes[node]['feature'][-5:-2] for node in subgraph.nodes]

    # Initialize a variable to track if any adjustments are needed
    adjustments_needed = True

    while adjustments_needed:
        adjustments_needed = False  # Assume no adjustments are needed initially

        # For each dimension (x, y, z)
        for dim in range(3):
            # Find the minimum coordinate value in this dimension
            min_val = min(coord[dim] for coord in all_coords)
            
            # If the minimum value is negative, adjust all coordinates in this dimension
            if min_val < 0:
                for coord in all_coords:
                    coord[dim] -= min_val  # Adjust so the minimum value becomes 0
                adjustments_needed = True  # Mark that adjustments were made

    # After all adjustments, all_coords now contains only non-negative coordinates
    return all_coords

def change_element_to_gold(subgraph, perfect=False):  #Not woring: I want to shift the separate cluster into the middle of the box
      
    features=subgraph.nodes[0]['feature']
    lat_list=[]
    if not lat_list:  # 假设所有节点的晶格信息相同，只需提取一次
        lat_list.extend(features[-26:-17] if perfect else features[-14:-5])  # 提取晶格信息
    lat=np.array(lat_list).reshape((3,3))
    #perf_coords.append(features[-17:-14] if perfect else features[-5:-2])  # 提取坐标信息
    elements_list = ["Li", "O", "Mg", "Al", "Ti", "V", "Mn", "Co", "Ni", "Zr"]
    
    # Step 1: Find the central atoms
    u, v, data_subgraph_lini_edge = find_center_atoms(subgraph)
    if u is None or v is None:
        return []  # No central atoms found

    u_coords = subgraph.nodes[u]['feature'][-17:-14] if perfect else subgraph.nodes[u]['feature'][-5:-2]
    v_coords = subgraph.nodes[v]['feature'][-17:-14] if perfect else subgraph.nodes[v]['feature'][-5:-2]
    u_element_index = np.argmax(subgraph.nodes[u]['feature'][:10])  # 假设前10维是独热编码
    u_element = elements_list[u_element_index]  # 获取元素种类
    v_element_index = np.argmax(subgraph.nodes[v]['feature'][:10])  # 假设前10维是独热编码
    v_element = elements_list[v_element_index]  # 获取元素种类
 
    # Step 3: Calculate the midpoint (center of the cluster)
    center = [(u + v) / 2 for u, v in zip(u_coords, v_coords)]
    u_coords_str = ", ".join([f"{u_coords0:.8f}" for u_coords0 in u_coords])
    v_coords_str = ", ".join([f"{v_coords0:.8f}" for v_coords0 in v_coords])
    print("*"*60)
    print(f"Extract from structure:{graph_name}")
    print(f"K-th neighbor:{k_nn}\nNode_number in subgraph:{node_number}")
    print(f"{u_element}-{u+1}: [{u_coords_str}]")
    print(f"{v_element}-{v+1}: [{v_coords_str}]")
    print(f"Exchange_E:{data_subgraph_lini_edge['delta_E']:.8f}")
    print("*"*60)
    coords = [subgraph.nodes[node]['feature'][-17:-14] if perfect else subgraph.nodes[node]['feature'][-5:-2] for node in subgraph.nodes]
    eles = [elements_list[np.argmax(subgraph.nodes[node]['feature'][:10])] if node!=u and node!=v else "Au" for node in subgraph.nodes]
    return coords,eles

 
def elements_and_counts(elements):
    # Use Counter to get the counts of each element
    counts = Counter(elements)
    # Separate the elements and their counts into two lists
    elements_list = list(counts.keys())
    counts_list = list(counts.values())
    return elements_list, counts_list

def create_out_file(out_type,lat,coords,species):
    output=[]

    if out_type=="xsf_std":
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" CONVVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords):
            output.append(f" {rl_aw(species[i])}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="xsf":
        output.append("DIM-GROUP\n")
        output.append("           3           1\n")
        output.append(" PRIMVEC\n")
        for i in lat:
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" PRIMCOORD\n")
        output.append(f"          {len(coords)}           1\n")
        for i,v in enumerate(coords):
            output.append(f" {species[i]}   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 

    elif out_type=="vasp":
        lattice_matrix=lat
        coords_frac=[]
        for coord in coords:
            car_coords_array = np.array(coord[:3])

            # 计算晶格矩阵的逆
            lattice_matrix_inv = np.linalg.inv(lattice_matrix)
            # 使用晶格矩阵的逆将笛卡尔坐标转换回分数坐标
            frac_coords_array = np.dot(car_coords_array,lattice_matrix_inv)
            coords_frac.append(frac_coords_array.tolist())
        elements, counts = elements_and_counts(species)
        output.append(f"VASP FILE {graph_name} {ind}\n")
        output.append("1.0\n")
        for i in lat:
            #print(i)
            output.append(f"   {i[0]:.10f}    {i[1]:.10f}    {i[2]:.10f}\n") 
        output.append(" "+"  ".join(elements)+"\n")
        output.append(" "+"  ".join([str(x) for x in counts])+"\n")
        output.append("Direct\n")
        for i,v in enumerate(coords_frac):
            output.append(f"   {v[0]:.10f}    {v[1]:.10f}    {v[2]:.10f}\n") 
        
    return output 

def create_output(out,name):
    f_out=open(f"{output_dir}/{name}","w+")
    for i in out:
        f_out.writelines(i)
    f_out.close()
    print(f"{output_dir}/{name} Created!")

subgraph=converted_subgraphs[ind]
#print([[u,v,data] for u,v,data in graph.edges(data=True)])
str_info=extract_lattice_and_coords(subgraph)
#print(str_info['orig_lat'])
#for i,v in enumerate(orig_coords):
    #print(orig_ele[i],v)

opt_lat=str_info['opt_lat']
opt_coords_orig=str_info['opt_coords']
opt_ele=str_info['opt_element']

if output_type=="xsf":
    out_filename=f"out_relax_{graph_name}_{ind}_k_{k_nn}.xsf"
elif output_type=="xsf_std":
    out_filename=f"out_relax_{graph_name}_{ind}_k_{k_nn}_standard.xsf"
elif output_type=="vasp":
    out_filename=f"out_relax_{graph_name}_{ind}_k_{k_nn}.vasp"

adjust_coords_to_positive

if args.perfect:
    perf_lat=str_info['perf_lat']
    perf_coords_orig=str_info['perf_coords']
    perf_ele=str_info['perf_element']
    output_perf_orig=create_out_file(output_type,perf_lat,perf_coords_orig,perf_ele)
    create_output(output_perf_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_perfect")) 
    #output_perf_orig=create_out_file(output_type,perf_lat,shift_cluster_to_box_center(subgraph,perfect=True),perf_ele)
    #create_output(output_perf_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_shift_center_perfect"))
    #output_perf_orig=create_out_file(output_type,perf_lat,adjust_coords_to_positive(subgraph,perfect=True),perf_ele)
    #create_output(output_perf_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_shift_center_perfect"))
    change_ele_coords,change_ele_eles=change_element_to_gold(subgraph,perfect=True)
    output_perf_orig=create_out_file(output_type,perf_lat,change_ele_coords,change_ele_eles)
    create_output(output_perf_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_mark_shift_lini_pair_perfect"))

output_opt_orig=create_out_file(output_type,opt_lat,opt_coords_orig,opt_ele)
create_output(output_opt_orig,out_filename) 
#output_opt_orig=create_out_file(output_type,opt_lat,shift_cluster_to_box_center(subgraph,perfect=False),opt_ele)
#create_output(output_opt_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_shift_center"))
#output_opt_orig=create_out_file(output_type,opt_lat,adjust_coords_to_positive(subgraph,perfect=False),opt_ele)
#create_output(output_opt_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_shift_center"))
change_ele_coords_opt,change_ele_eles_opt=change_element_to_gold(subgraph,perfect=False)
output_opt_orig=create_out_file(output_type,opt_lat,change_ele_coords_opt,change_ele_eles_opt)
create_output(output_opt_orig,out_filename.replace(f"{graph_name}",f"{graph_name}_mark_shift_lini_pair"))

################################################
####Creat html figure###########################
import plotly.graph_objs as go
import plotly.offline as py_offline
import plotly.io as pio
import math

color_map = {
    'Mn': 'rgba(128, 0, 128, 1)',  # 紫色
    'Ti': 'rgba(135, 206, 235, 1)', # 天蓝色
    'Co': 'rgba(0, 0, 139, 1)',     # 深蓝色
    'Ni': 'rgba(128, 128, 128, 1)', # 灰色
    'Li': 'rgba(0, 128, 0, 1)',     # 绿色
    'O': 'rgba(255, 0, 0, 1)'       # 红色
}

def calculate_node_positions(subgraph, center1, center2, k):
    """
    计算节点位置，使中心原子位于左右两侧，k阶邻节点位于第k环上。
    """
    positions = {}

    # 中心节点位置
    positions[center1] = (-1, 0)
    positions[center2] = (1, 0)

    def position_neighbors(node, level, angle_start, angle_end):
        """
        递归地为每层邻居节点分配位置。
        """
        if level > k:
            return

        neighbors = [n for n in subgraph.neighbors(node) if n not in positions]
        angle_step = (angle_end - angle_start) / max(len(neighbors), 1)

        for i, neighbor in enumerate(neighbors):
            angle = angle_start + i * angle_step
            x = level * math.cos(angle)
            y = level * math.sin(angle)
            positions[neighbor] = (x, y)
            position_neighbors(neighbor, level + 1, angle - angle_step / 2, angle + angle_step / 2)

    # 定位中心原子1的邻居
    position_neighbors(center1, 1, math.pi / 2, 3 * math.pi / 2)
    # 定位中心原子2的邻居
    position_neighbors(center2, 1, -math.pi / 2, math.pi / 2)

    return positions

def calculate_node_positions_for_centers(subgraph, center1, center2, k):
    positions = {}

    # 中心节点位置
    positions[center1] = (-1, 0)
    positions[center2] = (1, 0)

    def position_neighbors(node, level, angle_start, angle_end):
        if level > k:
            return

        neighbors = [n for n in subgraph.neighbors(node) if n not in positions and subgraph[node][n]['edge_type'] == 'original']
        angle_step = (angle_end - angle_start) / max(len(neighbors), 1)

        for i, neighbor in enumerate(neighbors):
            angle = angle_start + i * angle_step
            x = positions[node][0] + level * 0.5 * math.cos(angle)  # Adjust position based on the center node
            y = positions[node][1] + level * 0.5 * math.sin(angle)
            positions[neighbor] = (x, y)
            position_neighbors(neighbor, level + 1, angle - angle_step / 2, angle + angle_step / 2)

    # Positioning neighbors of center atoms
    position_neighbors(center1, 1, math.pi / 2, 3 * math.pi / 2)
    position_neighbors(center2, 1, -math.pi / 2, math.pi / 2)

    return positions

def calculate_node_positions_for_centers1(G, center1, center2, k):
    # Initialize the positions dictionary with center1 and center2
    positions = {center1: (-1, 0), center2: (1, 0)}
    labels = {center1: 'c1', center2: 'c2'}
    #print([key for key in positions],"Origin")       ######################debuging
    # Initialize the first layer
    #print(list(set(G.neighbors(center1))))
    #print(list(set(G.neighbors(center2))))
    exclusive_c1 = set([i for i in list(set(G.neighbors(center1))) if i not in list(set(G.neighbors(center2)))])-set([key for key in positions])
    exclusive_c2 = set([i for i in list(set(G.neighbors(center2))) if i not in list(set(G.neighbors(center1)))])-set([key for key in positions])
    shared = set([i for i in list(set(G.neighbors(center1))) if i in list(set(G.neighbors(center2)))])

    #print(exclusive_c1)
    #print(exclusive_c2)
    #print(shared)
    # Place the first layer neighbors
    start_angle = math.pi / 2
    end_angle = -math.pi / 2
    angle_step = (end_angle - start_angle) / (len(exclusive_c2) + 1)
    for i, node in enumerate(exclusive_c1, start=1):
        positions[node] = (-1 + math.sin(i * angle_step),math.cos(i * angle_step))
        labels[node] = 'c1'
    #print([[G.nodes[key]['element'],key] for key in positions],"C1-1")       ######################debuging  
    
    start_angle = -math.pi / 2
    end_angle = math.pi / 2
    angle_step = (end_angle - start_angle) / (len(exclusive_c2) + 1)
    for i, node in enumerate(exclusive_c2, start=1):
        positions[node] = (1 + math.sin(i * angle_step),math.cos(i * angle_step))
        labels[node] = 'c2'
    #print([[G.nodes[key]['element'],key] for key in positions],"C2-1")       ######################debuging
    # Place shared nodes on a straight line at y = 1
    shared_spacing = 2.0 / (len(shared) + 1)
    for i, node in enumerate(shared, start=1):
        positions[node] = (i * shared_spacing - 1, 1)
        labels[node] = 'share'
    #print([[G.nodes[key]['element'],key] for key in positions],"share-1")    ######################debuging
                
    # Function to place other layers
    def place_other_layers(nodes, level):
        if level > k:
            return
        
        next_nodes = set()
        neighbors = []
        
        c1_li=[]
        c2_li=[]
        share_li=[]
        
        for node in nodes: 
            neighbors.extend(set(G.neighbors(node)) - set(positions.keys()))
        neighbors=set(neighbors)
        #print(neighbors,"EE")       ######################debuging
        
        for node in neighbors:
            #################################################################################################
            #print("A",node,"nb",[oo for oo in G.neighbors(node)])
            #print("*"*20,node,[labels[o] for o in G.neighbors(node) if o in [key for key in positions]])
    
            if all(labels[element] == 'c1' for element in G.neighbors(node) if element in [key for key in positions]):
                c1_li.append(node)

            elif all(labels[element] == 'c2' for element in G.neighbors(node) if element in [key for key in positions]):
                c2_li.append(node)

            else:  # For shared nodes
                share_li.append(node)
        
        for i, neighbor in enumerate(c1_li, start=1):
            start_angle = math.pi / 2
            end_angle = -math.pi / 2
            angle_step = (end_angle - start_angle) / (len(c1_li) + 1)
            #print([G.nodes[neighbor]['element'],neighbor],neighbor,'c1')
            positions[neighbor] = (-1 + level * math.sin(i * angle_step),level * math.cos(i * angle_step))
            labels[neighbor] = 'c1'
        for i, neighbor in enumerate(c2_li, start=1):            
            start_angle = -math.pi / 2
            end_angle = math.pi / 2
            angle_step = (end_angle - start_angle) / (len(c2_li) + 1)
            #print([G.nodes[neighbor]['element'],neighbor],neighbor,'c2')
            positions[neighbor] = (1 + level * math.sin(i * angle_step),level * math.cos(i * angle_step))
            labels[neighbor] = 'c2'
        for i, neighbor in enumerate(share_li, start=1):
            shared_spacing = 2.0 / (len(share_li) + 1)
            #print([G.nodes[neighbor]['element'],neighbor],neighbor,'share')
            positions[neighbor] = (i * shared_spacing - 1, level)
            labels[neighbor] = 'share'
        
        #print(next_nodes,"AA")
        #print([[G.nodes[key]['element'],key] for key in neighbors],"BB")
        #print(c1_li)
        #print(c2_li)
        #print(share_li)
        next_nodes = set(c1_li) | set(c2_li) | set(share_li) | nodes        
        place_other_layers(next_nodes, level + 1)

    # Place other layers
    all_first_layer_nodes = exclusive_c1 | shared | exclusive_c2
    #print(all_first_layer_nodes,"ALL")
    place_other_layers(all_first_layer_nodes, 2)

    return positions, labels




# 从子图中找出两个中心原子
def find_center_atoms(subgraph):
    # 假设中心原子是连接li_ni_edge的两个原子
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            return u, v
    return None, None

def draw_subgraph_with_colored_nodes_and_energy(subgraph, pos, center1, center2, color_map):
    edge_x = []
    edge_y = []
    red_edge_x = []
    red_edge_y = []

    # 初始化交换能量变量
    exchange_energy = None
    # 定义结构名称读取参数
    str_name = None
    # 处理边
    for edge in subgraph.edges():
        #print(pos[edge[0]])
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if edge == (center1, center2) or edge == (center2, center1):
            red_edge_x.extend([x0, x1, None])
            red_edge_y.extend([y0, y1, None])
            exchange_energy = subgraph.edges[edge].get('delta_E', 'N/A')  # 获取交换能量
            str_name = subgraph.edges[edge].get('str_name', 'N/A')
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    # 边的线条
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    red_edge_trace = go.Scatter(x=red_edge_x, y=red_edge_y, line=dict(width=1, color='red'), hoverinfo='none', mode='lines')

    # 节点的坐标、标签和颜色
    node_x = [pos[node][0] for node in subgraph.nodes()]
    node_y = [pos[node][1] for node in subgraph.nodes()]
    node_text = [f"{node+1}<br>({subgraph.nodes[node]['original_element']}){[[n+1,subgraph.nodes[n]['element']] for n in subgraph.neighbors(node)]}" for node in subgraph.nodes()]
    node_colors = [color_map[subgraph.nodes[node]['original_element']] for node in subgraph.nodes()]

    # 节点的标记
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', hovertext=node_text,
                            marker=dict(color=node_colors, size=10, line_width=2))

    # 构造标题
    title = f"Structure:{str_name} K={k_nn} Nodes:{node_number} Center Atom:{center1+1}#{subgraph.nodes[center1]['original_element']} - {center2+1}#{subgraph.nodes[center2]['original_element']} Exchange Energy: {exchange_energy}"

    # 创建图形
    fig = go.Figure(data=[edge_trace, red_edge_trace, node_trace], layout=go.Layout(title=title, titlefont_size=16, showlegend=False, 
                            hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(text="Subgraph Visualization", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    #fig.show()
    py_offline.plot(fig, filename=f'{output_fig_dir}/{graph_name}_{ind}_k_{k_nn}.html', auto_open=False)
    print(f'{output_fig_dir}/{graph_name}_{ind}_k_{k_nn}.html saved!')


    node_text2 = [f"{node+1} ({subgraph.nodes[node]['original_element']})" for node in subgraph.nodes()]
    node_trace2 = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text2,
                            textfont=dict(size=png_font),  # 调整字体大小为12，或根据需要进行调整    
                            textposition='top center',  # 文本位置在标记的正上方
                            marker=dict(color=node_colors, size=10, line_width=2))

    # 构造标题
    #title = f"Structure:{str_name} K={k_nn} Nodes:{node_number} Center Atom:{center1+1}#{subgraph.nodes[center1]['original_element']} - {center2+1}#{subgraph.nodes[center2]['original_element']} Exchange Energy: {exchange_energy}"

    # 创建图形
    fig2 = go.Figure(data=[edge_trace, red_edge_trace, node_trace2], layout=go.Layout(title=title, titlefont_size=16, showlegend=False, 
                            hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(text="Subgraph Visualization", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    pio.write_image(fig2, f'{output_fig_dir}/{graph_name}_{ind}_k_{k_nn}_{resolution[0]}_{resolution[1]}.png', width=resolution[0]*100, height=resolution[1]*100)
    print(f'{output_fig_dir}/{graph_name}_{ind}_k_{k_nn}_{resolution[0]}_{resolution[1]}.png saved!')
    if os.path.isfile("show_fig.py"):
        import subprocess as sb
        sb.Popen(["python","show_fig.py",f'{output_fig_dir}/{graph_name}_{ind}_k_{k_nn}_{resolution[0]}_{resolution[1]}.png'])
        
# 使用这个函数绘制子图
# draw_subgraph_with_colored_nodes_and_energy(subgraph, pos, center1, center2, color_map)


# 假设 subgraphs[0] 是我们要绘制的子图
'''
subgraphs_i=subgraphs[0]
center1, center2 = find_center_atoms(subgraphs_i)
if center1 is not None and center2 is not None:
    pos = calculate_node_positions(subgraphs_i, center1, center2, k=2)
    # 构造标题
    title = f"{center1+1}#{subgraph_i.nodes[center1]['element']} - {center2+1}#{subgraphs_i.nodes[center2]['element']}: 交换能量"
    draw_subgraph_with_calculated_positions(subgraphs[0], pos, title)
else:
    print("无法确定中心原子")
'''
# Assuming subgraphs[0] is the subgraph we want to draw
if create_figs:
    center1, center2 = find_center_atoms(subgraph)
    #print(center1,center2)
    if center1 is not None and center2 is not None:
        #pos = calculate_node_positions_for_centers_modified(subgraph_i, center1, center2, k=2)
        pos,lables = calculate_node_positions_for_centers1(subgraph, center1, center2, k=k_nn+1)
        #print(pos,lables)
        draw_subgraph_with_colored_nodes_and_energy(subgraph, pos, center1, center2, color_map)
    
    else:
        print("Unable to determine center atoms")
