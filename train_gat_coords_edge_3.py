#增加了local节点性质提取，增加了node Feature归一化

#提取数据
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
#from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data, DataLoader
import sqlite3
import json
import networkx as nx
import random
import argparse

parser = argparse.ArgumentParser(description='CGAT Train Code, it would generate a file with your training results')
parser.add_argument("--debug",'-d',action='store_true',default=False,help='debug and test the first three steps')
args = parser.parse_args()

#db_path = 'C:\\Users\\xiety\\Desktop\\NN\GNN\\final_used_str_establish\\subgraphs_4_v3_valance.db'
db_path = './subgraphs_k_neighbor_5_gnn_53d_feature_train.db'

split_ratio=0.9
batch_size=40

lr_rate=0.08 
weight_decay_value=1e-5  # 添加L2正则化

num_epochs = 20000  # 根据需要设定训练的轮数
early_stopping_patience =  40  # 早停的耐心值

k_nn=2   #pooling用到的邻居,但是我们GAT没有pooling
k_hop_attention=5  #Attention额外添加边的邻居

gat_eps=1e-5    
gat_heads=12       #multi-head-attention number

opt_dis=True
# 选择一个种子值
seed = 411


debug=args.debug   #debug means not create anything

if debug==1:
    print("Debug model on, there will train 3 epochs for debug!")
    num_epochs=3

def set_seed(seed):
    #"""设置全局随机种子以确保实验的可复现性"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

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
    
    random.shuffle(subgraphs)  # 随机打乱数据
    #np.random.seed(42)
    split_idx = int(len(subgraphs) * split_ratio)

    train_graphs = subgraphs[:split_idx]
    test_graphs = subgraphs[split_idx:]
    
    return train_graphs, test_graphs

all_graphs  = load_data_from_db(db_path)
train_graphs, test_graphs = create_networkx_graphs(all_graphs)
print(len(all_graphs),len(train_graphs),len(test_graphs))
print(train_graphs[0])

#转化数据，
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
#from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import json
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def get_neighbors_xty(graph, node, depth):
    get_nodes = [node]
    node_buffer = []
    for i in range(depth):
        for node in get_nodes:
            for neighbor in graph.neighbors(node):
                if graph.edges[node, neighbor]['edge_type'] == 'original':
                    node_buffer.append(neighbor)
        node_buffer = list(set(node_buffer))
        get_nodes.extend(node_buffer)
    get_nodes.remove(node)
    return set(get_nodes)


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



def networkx_to_pyg_graph(subgraph):
    
    # Convert to zero-based indexing重新标记节点索引：为了确保edge_index中的索引是零起始和连续的，我添加了一个映射来重新标记您的networkx图中的节点。这个映射将每个节点映射到一个新的索引（从0开始）
    mapping = {node: i for i, node in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, mapping)

    # 计算每条边的距离并存储
    #perf_lat.extend(features[-26:-17])  # 提取优化前晶格信息
    #opt_lat.extend(features[-14:-5])  # 提取优化后晶格信息
    #perf_coords.append(features[-17:-14])  # 提取优化前坐标信息
    #opt_coords.append(features[-5:-2])  # 提取优化后坐标信息

    if opt_dis==True:
        range1_lat_coord=-14
        range2_lat_coord=-2
    elif opt_dis==False:
        range1_lat_coord=-26
        range2_lat_coord=-14

    edge_distances = []
    for u, v, data in subgraph.edges(data=True):
        u_vec=subgraph.nodes[u]['feature'][range1_lat_coord:range2_lat_coord]  
        v_vec=subgraph.nodes[v]['feature'][range1_lat_coord:range2_lat_coord]
        distance = calculate_minimum_distance(u_vec,v_vec)
        edge_distances.append([distance])
    
    #print(edge_distances)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float)
    
    # 提取节点特征
    node_features = [subgraph.nodes[node]['feature'][:27]for node in subgraph.nodes()]
    # 创建归一化器实例
    scaler = MinMaxScaler()
    # 拟合数据并转换
    node_features_normalized = scaler.fit_transform(node_features)
    x = torch.tensor(node_features_normalized, dtype=torch.float)
    #x = torch.tensor(node_features, dtype=torch.float)

    # 提取边索引和边类型
    edge_indices = []
    edge_types = []
    for u, v, data in subgraph.edges(data=True):
        edge_indices.append((u, v))
        # 假设 'original' 边为类型 0，'li_ni_edge' 为类型 1
        edge_type = 1 if data.get('edge_type') == 'li_ni_edge' else 0
        edge_types.append(edge_type)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    # 计算k跳邻居并标记节点

    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            k_neighbors_u = get_neighbors_xty(subgraph, u, k_nn)
            k_neighbors_v = get_neighbors_xty(subgraph, v, k_nn)
            collected_nodes = k_neighbors_u.union(k_neighbors_v)       

            


    def get_k_hop_neighbors(graph, node, depth=k_hop_attention):
        get_nodes=[node]
        node_buffer=[]
        for i in range(1,depth+1):
            #print(i,"depth")
            for node in get_nodes:
                #print(node,"node")
                for neighbor in graph.neighbors(node):
                    if graph.edges[node,neighbor]['edge_type']=='original':
                        #if depth==1:
                            #print(neighbor,"select_neighbor")
                        node_buffer.append(neighbor)
            node_buffer=list(set(node_buffer))
            get_nodes.extend(node_buffer)
            
        return set(get_nodes)
            
    # 查找包含能量信息的 'li_ni_edge'
    exchange_energy = None
    li_ni_pairs=None
    additional_edge_indices = []  # 存储额外边的索引
    additional_edge_distances = []  # 存储额外边的距离属性
    printed_neighbors_info = False
    
    for u, v, data in subgraph.edges(data=True):
        if data.get('edge_type') == 'li_ni_edge':
            exchange_energy = data.get('delta_E', None)
            #li_ni_pair=[u,v]# 保存 li_ni_edge 边两端节点的索引
            li_ni_pairs = [[u, v]]
            
            # 获取u和v的k近邻
            k_hop_neighbors_u = get_k_hop_neighbors(subgraph, u, k_hop_attention)
            k_hop_neighbors_v = get_k_hop_neighbors(subgraph, v, k_hop_attention)
            
            # 对每个节点，添加到u和v的额外边
            for node in subgraph.nodes():
                if node != u and node != v:  # 排除u和v自身
                    # 添加到u的边
                    if node in k_hop_neighbors_u:
                        u_vec = subgraph.nodes[u]['feature'][range1_lat_coord:range2_lat_coord]
                        node_vec = subgraph.nodes[node]['feature'][range1_lat_coord:range2_lat_coord]
                        distance = calculate_minimum_distance(u_vec, node_vec)
                        additional_edge_indices.append([node, u])
                        additional_edge_distances.append([distance])

                    # 添加到v的边
                    if node in k_hop_neighbors_v:
                        v_vec = subgraph.nodes[v]['feature'][range1_lat_coord:range2_lat_coord]
                        node_vec = subgraph.nodes[node]['feature'][range1_lat_coord:range2_lat_coord]
                        distance = calculate_minimum_distance(v_vec, node_vec)
                        additional_edge_indices.append([node, v])
                        additional_edge_distances.append([distance])
            # 打印新增邻居的个数
            #if not printed_neighbors_info:
                #print(f"New neighbors for u: {len(k_hop_neighbors_u)}")
                #print(f"New neighbors for v: {len(k_hop_neighbors_v)}")
                #printed_neighbors_info = True
            break

    # 检查是否找到了带有能量信息的边
    if exchange_energy is None:
        return None

    # 将能量信息转换为 PyTorch 张量
    y = torch.tensor([exchange_energy], dtype=torch.float)
    li_ni_pairs_tensor = torch.tensor(li_ni_pairs, dtype=torch.long)
    k_neighbors=[list(collected_nodes)]
    k_neighbors_tensor = torch.tensor(k_neighbors, dtype=torch.long)
    u_nb=torch.tensor([list(k_neighbors_u)],dtype=torch.long)
    v_nb=torch.tensor([list(k_neighbors_v)],dtype=torch.long)
    
    # 记录每个图的节点数量
    num_nodes = len(subgraph.nodes())

    # 添加额外边的属性
    edge_distances.extend(additional_edge_distances)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float)

    # 添加额外边的索引
    edge_indices.extend(additional_edge_indices)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    # 创建 PyTorch Geometric Data 对象
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, edge_attr=edge_attr, y=y, li_ni_pairs=li_ni_pairs_tensor, k_neighbors=k_neighbors_tensor, u_nb=u_nb, v_nb=v_nb, num_nodes=num_nodes)
    return data

# 转换所有训练和测试子图

###SHIT MOUTAIN SEE BELOW###

#train_dataset = [networkx_to_pyg_graph(subgraph) for subgraph in train_graphs if networkx_to_pyg_graph(subgraph) is not None]
#test_dataset = [networkx_to_pyg_graph(subgraph) for subgraph in test_graphs if networkx_to_pyg_graph(subgraph) is not None]
#train_dataset = []
#test_dataset = []  
#print("Create train graphs...")
#for i,subgraph in enumerate(train_graphs):
#    print("["+"#"*int(i/len(train_graphs)*60)+"-"*int((1-i/len(train_graphs))*60)+"]"+f"{(i+1)/len(train_graphs)*100:.2f}%"+f"   {(i+1)}/{len(train_graphs)}","\r",end='')
#    if networkx_to_pyg_graph(subgraph) is not None:
#        train_dataset.append(networkx_to_pyg_graph(subgraph))
#print("\nCreate test graphs...")
#for i,subgraph in enumerate(test_graphs):
#    print("["+"#"*int(i/len(test_graphs)*60)+"-"*int((1-i/len(test_graphs))*60)+"]"+f"{(i+1)/len(test_graphs)*100:.2f}%"+f"   {(i+1)}/{len(test_graphs)}","\r",end='')
#    if networkx_to_pyg_graph(subgraph) is not None:
#        test_dataset.append(networkx_to_pyg_graph(subgraph))

###SHIT MOUTAIN SEE ABOVE(I don't know when the below perfect code would colapese so I keep the shit moutain)###

#import torch_geometric
from concurrent.futures import ProcessPoolExecutor, as_completed
import networkx as nx
from time import sleep
def process_subgraph(subgraph):
    # 假设networkx_to_pyg_graph是你的函数，用于转换图的格式
    pyg_graph = networkx_to_pyg_graph(subgraph)
    return pyg_graph

def print_progress(done, total):
    #假设trains(done, total):
    percent_done = done / total * 100
    bar_length = int(percent_done / 100 * 60)
    bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
    print(bar, "\r", end='')

def convert_graphs(graphs):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_subgraph, subgraph): i for i, subgraph in enumerate(graphs)}
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

    return [graph for graph in converted_graphs if graph is not None]
#_graphs和test_graphs已经被定义
print("Create train graphs...")
train_dataset = convert_graphs(train_graphs)
print("\nCreate test graphs...")
test_dataset = convert_graphs(test_graphs)
print()

print(len(train_dataset),len(test_dataset))

# Usual train

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Whole dataset train

#train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Normalized train

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 假设你已经有了一个torch_geometric的Dataset对象dataset
# 设置 DataLoader的worker_init_fn
g_data = torch.Generator()
g_data.manual_seed(0)  # 为了确保每次都能复现相同的数据加载顺序，设置一个固定的种子


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker,generator=g_data)#num_workers是并行处理数据的子进程数量
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker,generator=g_data)


# 获取前三个样本
first_three_samples = []
for i, data in enumerate(train_loader):
    if i >= 1: 
        break
    first_three_samples.append(data)

# 检查每个样本
for i, data in enumerate(first_three_samples):
    
    # 模拟模型的 forward 过程
    x, edge_index, edge_type, edge_attr, batch, li_ni_indices, k_neighbors, u_nb, v_nb, num_nodes = (
        data.x, data.edge_index, data.edge_type, data.edge_attr, data.batch, data.li_ni_pairs, 
        data.k_neighbors, data.u_nb, data.v_nb, data.num_nodes
    )

    batch_size = batch.max() + 1
    num_nodes_per_graph = num_nodes // batch_size  # 每个图的节点数量
    start_idx = torch.arange(0, batch_size * num_nodes_per_graph, num_nodes_per_graph, device=x.device)
    # 计算每条边属于哪个图
    graph_ids = torch.tensor([i for i in range(batch_size)])


    # 调整li_ni_indices的索引
    adjusted_li_ni_indices = li_ni_indices + start_idx[graph_ids].unsqueeze(1)

print("Node X shape:", x.shape)
print("Edge_index shape:", edge_index.shape)  # 打印edge_index的维度和内容
print("Train_dataset:",len(train_dataset))
print("Test_dataset:",len(test_dataset))
print("Train_loader:",len(train_loader))
print("Test_loader:",len(test_loader))
print(train_dataset[0])
print(f"batch_size: {batch_size}")

class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=None, concat=False):
        super(CustomGATConv, self).__init__(aggr='add')  # 使用加法聚合
        #self.linear = torch.nn.Linear(in_channels, out_channels)  # 用于特征转换
        #self.att = torch.nn.Parameter(torch.Tensor(1, 2 * out_channels))  # 注意力系数参数
        
        self.concat=concat       
        
        if heads==None:
            heads=1
        
        self.heads = heads
        
        # 为每个头创建一个独立的线性变换层和注意力向量
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(in_channels, out_channels) for _ in range(heads)
        ])
        self.atts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(1, 2 * out_channels)) for _ in range(heads)
        ])        
        
        self.reset_parameters()

    #def reset_parameters(self):
    #    torch.nn.init.xavier_uniform_(self.linear.weight)
    #    torch.nn.init.xavier_uniform_(self.att)
    #def forward(self, x, edge_index, edge_attr):
    #    x = self.linear(x)
    #    self.x=x
    #    return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def reset_parameters(self):
        for linear in self.linears:
            torch.nn.init.xavier_uniform_(linear.weight)
        for att in self.atts:
            torch.nn.init.xavier_uniform_(att)
    
    def forward(self, x, edge_index, edge_attr):
        # 分别对每个头计算注意力和转换后的特征
        x_heads = [linear(x) for linear in self.linears]
        
        # 将每个头的结果分别传递给propagate并聚合
        outs = [self.propagate(edge_index, x=x_head, edge_attr=edge_attr, head_idx=i)
                for i, x_head in enumerate(x_heads)]
        
        # 聚合所有头的结果（这里使用拼接）
        if self.concat==False:
            self.x = torch.stack(outs, dim=0).mean(dim=0)
        else:    
            self.x = torch.cat(outs, dim=-1)
        
        return self.x
    
    def message(self, edge_index_i, x_i, x_j, edge_attr, head_idx):
        # 使用对应头的注意力参数
        attention = self.atts[head_idx]
        x_j_cat = torch.cat([x_i, x_j], dim=1)
        attention_score = torch.matmul(x_j_cat, attention.t())
        attention_score = F.leaky_relu(attention_score)

        edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)**2
        edge_attr_inv_sq = edge_attr_inv_sq.view(-1, 1)

        weighted_message = x_j * edge_attr_inv_sq
        x_j_transformed = attention_score * weighted_message
        return x_j_transformed
    
    def message_gcn(self, x_j, edge_attr):
        # 现在我们只使用边属性进行加权，不计算注意力分数
        edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)
        return x_j * edge_attr_inv_sq.view(-1, 1)    
    
    #def message(self, edge_index_i, x_i, x_j, edge_attr):
        # 计算注意力系数        
        #print("x_j.shape=",x_j.shape)
        #print("x_i.shape=",x_i.shape)        
    #    x_j_cat = torch.cat([x_i, x_j], dim=1)  # 将源节点和目标节点的特征拼接        
        #print("x_j_cat.shape=",x_j_cat.shape)        
    #    attention = torch.matmul(x_j_cat, self.att.t())        
        #print("self.att.t().shape=",self.att.t().shape)        
    #    attention = F.leaky_relu(attention)
        # 将edge_attr用作门控机制
    #    edge_attr_inv_sq = 1.0 / (edge_attr + 1e-7)**2  # 依然计算距离的平方倒数
    #    edge_attr_inv_sq = edge_attr_inv_sq.view(-1, 1)  # 确保尺寸匹配
        #print("edge_att.shape=",edge_attr_inv_sq.shape)        
        # 使用edge_attr加权信息
    #    weighted_message = x_j * edge_attr_inv_sq  # 使用edge_attr调整信息流的权重
    #    x_j_transfrom = attention * weighted_message
        #print("x_j_transfrom.shape",x_j_transfrom.shape)
        #print("self.x.shape",self.x.shape)
        # 应用注意力系数和门控后的信息加权结果
    #    return x_j_transfrom   
    
    def update(self, aggr_out):
        #print("aggr_out.shape=",aggr_out.shape)
        #aggr_out+=self.x
        #aggr_out=F.relu(aggr_out)
        return aggr_out
        #return F.relu(aggr_out)
    
    

class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GAT,self).__init__()
        #super(GAT, self).__init__()
        # 定义GAT卷积层
        self.conv1 = CustomGATConv(num_node_features, num_node_features, heads=gat_heads, concat=True)
        self.conv2 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv3 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv4 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv5 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv6 = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=True)
        self.conv_f = CustomGATConv(num_node_features * gat_heads, num_node_features, heads=gat_heads, concat=False)
        
        self.bn1 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  # BN for conv1 output,eps为了防止除以0
        self.bn2 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  # BN使用时需要在激活函数之前 
        self.bn3 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  
        self.bn4 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)
        self.bn5 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)  
        self.bn6 = torch.nn.BatchNorm1d(num_node_features * gat_heads, eps=gat_eps)
        
        self.hidden_layer1 = torch.nn.Linear(num_node_features , num_node_features*2 )
        self.hidden_layer2 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        self.hidden_layer3 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        self.hidden_layer4 = torch.nn.Linear(num_node_features*2 , num_node_features*2 )
        
        self.bn_hidden1 = torch.nn.BatchNorm1d(num_node_features , eps=gat_eps)
        self.bn_hidden2 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        self.bn_hidden3 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        self.bn_hidden4 = torch.nn.BatchNorm1d(num_node_features * 2, eps=gat_eps)
        
        self.predictor = torch.nn.Linear(num_node_features*2, 1)

        self.first_pair_neighbors_printed = False
        
        self.num_element_types = 10  # 假定有10种不同的元素
        self.embedding_dim = 10  # 假定每个元素的嵌入维度为10
        self.element_embedding = torch.nn.Embedding(self.num_element_types, self.embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # 重置GAT卷积层权重
        for conv in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv_f]:
            conv.reset_parameters()
        
        # 重置全连接层权重
        for linear in [self.hidden_layer1, self.hidden_layer2, self.hidden_layer3, self.predictor]:
            torch.nn.init.xavier_uniform_(linear.weight)
            linear.bias.data.fill_(0)
        
        
    def forward(self, x, edge_index, edge_attr, batch, li_ni_indices, k_neighbors, u_nb, v_nb, num_nodes):
        # 应用GAT卷积层
        x1 = self.conv1(x, edge_index, edge_attr) 
        x1 = F.relu(self.bn1(x1))
        #x1 = F.relu(x1)
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = F.relu(self.bn2(x2))
        #x2 = F.relu(x2)
        x3 = self.conv3(x2, edge_index, edge_attr) + self.conv3(x1, edge_index, edge_attr)  
        x3 = F.relu(self.bn3(x3))
        #x3 = F.relu(x3)
        x4 = self.conv4(x3, edge_index, edge_attr)   
        x4 = F.relu(self.bn4(x4))
        x5 = self.conv5(x4, edge_index, edge_attr)
        x5 = F.relu(self.bn5(x5))
        x6 = self.conv6(x5, edge_index, edge_attr) + self.conv6(x4, edge_index, edge_attr)  
        x6 = F.relu(self.bn6(x6))
        x_f = self.conv_f(x4, edge_index, edge_attr)  
        # 计算每个图的起始索引
        batch_size = batch.max() + 1
        # 假设每个图的节点数量是固定的
        num_nodes_per_graph = num_nodes/batch_size  # 每个图的节点数量
        
        start_idx = torch.arange(0, batch_size * num_nodes_per_graph, num_nodes_per_graph, device=x.device)
        # 计算每条边属于哪个图
        graph_ids = torch.tensor([i for i in range(batch_size)])
        #print("Graph IDs for each edge:", graph_ids)

        # 调整li_ni_indices的索引
        adjusted_li_ni_indices = li_ni_indices + start_idx[graph_ids].unsqueeze(1)
        adjusted_u_nb = u_nb + start_idx[graph_ids].unsqueeze(1)
        adjusted_v_nb = v_nb + start_idx[graph_ids].unsqueeze(1)
        adjusted_k_neighbors = k_neighbors + start_idx[graph_ids].unsqueeze(1)
        # 计算每个li_ni_edge边两端的k近邻的平均特征
        k_hop_features = []
        for i, (li, ni) in enumerate(adjusted_li_ni_indices):
            # 获取li和ni的k跳邻居
            li_neighbors = adjusted_u_nb[i]
            ni_neighbors = adjusted_v_nb[i]
            k_neighbors_i = adjusted_k_neighbors[i]
            # 打印第一个li_ni对的邻居个数
            if not self.first_pair_neighbors_printed:
                print("Li-Ni_indices:",li,ni)
                print("Number of neighbors for li:", len(li_neighbors),li_neighbors)
                print("Number of neighbors for ni:", len(ni_neighbors),ni_neighbors)
                print("K_neighbors:",len(k_neighbors_i),k_neighbors_i)
                self.first_pair_neighbors_printed = True

            # 获取邻居节点的特征并计算平均
            #li_neighbor_features = x_f[li_neighbors.long()].mean(dim=0)
            #ni_neighbor_features = x_f[ni_neighbors.long()].mean(dim=0)
            li_neighbor_features = x_f[li.long()]
            ni_neighbor_features = x_f[ni.long()]
            
            #聚合li和ni的邻居特征
            edge_features = (li_neighbor_features + ni_neighbor_features) / 2
            
            #li_tensor = li_neighbor_features.clone().detach()
            #ni_tensor = ni_neighbor_features.clone().detach()
            # Concatenate along a new dimension (e.g., dim=0)
            #edge_features = torch.cat((li_tensor, ni_tensor), dim=0)
 
            
            k_hop_features.append(edge_features)
        
        k_hop_features_tensor = torch.stack(k_hop_features, dim=0)
        # 使用预测器进行预测
        # 使用预测器进行预测
        # 预激活：先应用BN，然后是ReLU激活函数
        x_hidden1 = self.bn_hidden1(k_hop_features_tensor)
        x_hidden1 = F.relu(x_hidden1)
        x_hidden1 = self.hidden_layer1(x_hidden1)
        # 注意，这里没有在ReLU和第一个全连接层之间应用BN

        # 对于第二个隐藏层，我们也采用预激活的方式
        x_hidden2 = self.bn_hidden2(x_hidden1)
        x_hidden2 = F.relu(x_hidden2)
        x_hidden2 = self.hidden_layer2(x_hidden2) + x_hidden1
        # 注意，在x_hidden2的计算中加入了x_hidden1进行残差连接

        # 第三个隐藏层同样使用预激活方式
        x_hidden3 = self.bn_hidden3(x_hidden2)
        x_hidden3 = F.relu(x_hidden3)
        x_hidden3 = self.hidden_layer3(x_hidden3) + x_hidden2
        # 同样，在x_hidden3的计算中加入了x_hidden2进行残差连接

        x_hidden4 = self.bn_hidden4(x_hidden3)
        x_hidden4 = F.relu(x_hidden4)
        x_hidden4 = self.hidden_layer4(x_hidden4) + x_hidden3
        # 最后应用预测层得到最终的输出
        energy_prediction = self.predictor(x_hidden4)
        #energy_prediction = self.predictor(k_hop_features_tensor)
        return energy_prediction
        
        
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
# [您的模型初始化和其他设置] 

# 用于记录损失和R²值
train_losses = []
test_losses = []
train_r2_scores = []
test_r2_scores = []

#训练神经网络
num_node_features = len(train_dataset[0].x[0])
num_relations = 2  # 'original' 和 'li_ni_edge'
#model = RGCN(num_node_features, num_relations)
model = GAT(num_node_features)

#model = RGCN(num_node_features, num_relations)
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate, weight_decay=weight_decay_value) # 添加L2正则化
criterion = torch.nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, 'min')  # 学习率调整策略

min_loss = float('inf')
early_stopping_counter = 0

start_time_total = time.time()
round_time = 0

# 训练模型
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    predictions, labels = [], []
    
    # 收集的训练和测试预测值及标签:batch
    train_predictions = []  
    test_predictions = []   
    train_labels = []       
    test_labels = [] 
    round_nn=0
    for data in train_loader:
        print(f"train: {round_nn}/{len(train_loader)}","\r",end='')
        optimizer.zero_grad()
        #RGCN #GAT
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.li_ni_pairs, data.k_neighbors, data.u_nb, data.v_nb, data.num_nodes)
       
        # 检查模型输出是否包含NaN
        if torch.isnan(out).any():
            print(f"NaN detected in output at epoch {epoch}")
            break  # 退出训练循环
        #out = model(data.x, data.edge_index, data.batch, data.li_ni_pairs)
        train_predictions.extend(out.detach().squeeze().tolist())
        train_labels.extend(data.y.tolist())
        
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度裁剪，如果没有就会梯度爆炸
        
        optimizer.step()
        total_loss += loss.item()
        predictions.extend(out.detach().squeeze().tolist())
        labels.extend(data.y.tolist())
        round_nn+=1
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)
    train_r2 = r2_score(labels, predictions)
    train_r2_scores.append(train_r2)

    # 测试模型
    model.eval()
    test_loss = 0
    with torch.no_grad():
        round_nn=1
        for data in test_loader:
            print(f"test: {round_nn}/{len(test_loader)}","\r",end='')
            #RGCN #GAT
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.li_ni_pairs, data.k_neighbors, data.u_nb, data.v_nb, data.num_nodes)

            #out = model(data.x, data.edge_index, data.batch, data.li_ni_pairs)
            test_predictions.extend(out.squeeze().tolist())
            test_labels.extend(data.y.tolist())
            
            test_loss += criterion(out.squeeze(), data.y).item()
            round_nn+=1
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    
    test_r2 = r2_score(test_labels, test_predictions)
    test_r2_scores.append(test_r2)

    scheduler.step(test_loss)  # 调整学习率

    # 早停检查
    if test_loss < min_loss:
        min_loss = test_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    end_time=time.time()
    run_time = round(end_time-start_time)
    run_time_total = round(end_time - start_time_total)
    hour = run_time//3600
    minute = (run_time-3600*hour)//60
    second = run_time-3600*hour-60*minute
    hour_total = run_time_total//3600
    minute_total = (run_time_total-3600*hour_total)//60
    second_total = run_time_total-3600*hour_total-60*minute_total
    round_time+=1
    #print_stuff=f"Running: {hour}小时{minute}分钟{second}秒"
    print(f'Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train R²: {train_r2}, Test R²:{test_r2}')
    print(f"**Time cost: this round:00:{minute}:{second}, Total:{hour_total}:{minute_total}:{second_total},round:{round_time}")
print(f'Train R²: {train_r2},Test R²: {test_r2}')


r2=r2_score(test_labels, test_predictions)
mse=np.sum(np.square(np.array(test_predictions) - np.array(test_labels))) / len(test_labels)
mae=np.sum(np.abs(np.array(test_predictions) - np.array(test_labels))) / len(test_labels)
print("R2 on GNN:",r2)
print("MSE on GNN:",mse)
print("MAE on GNN:",mae)

if debug==0:

    # 保存模型的状态字典
    print("Saving models...")
    torch.save(model.state_dict(), f'model_GAT_train-valence-Resnet-aggregate_k{k_hop_attention}-FC3_4_heads_{gat_heads}_seed_{seed}_R2_{r2:.3f}_opt_{opt_dis}.pth')
    
    # 绘制训练和测试的损失曲线
    print("Drawing losses...")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss:{train_losses[-1]:.4f}')
    plt.plot(test_losses, label=f'Test Loss:{test_losses[-1]:.4f}')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE:eV^2)')
    plt.ylim(-0.2,2)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_r2_scores, label=f'Train R² Score:{train_r2_scores[-1]:.4f}')
    plt.plot(test_r2_scores, label=f'Test R² Score:{test_r2_scores[-1]:.4f}')
    plt.ylim(-0.2,1)
    plt.title('R² Score over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('loss_10_4.png',dpi=900)
    
    #画残差图
    print("Drawing residuals...")
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    
    # 函数用于计算95%置信区间
    def confidence_interval(data):
        std_dev = np.std(data)
        mean = np.mean(data)
        ci_lower = mean - 1.96 * std_dev
        ci_upper = mean + 1.96 * std_dev
        return ci_lower, ci_upper
    
    # 计算残差
    train_residuals = np.array(train_labels) - np.array(train_predictions)
    test_residuals = np.array(test_labels) - np.array(test_predictions)
    
    # 计算测试残差的置信区间
    ci_lower, ci_upper = confidence_interval(test_residuals)
    
    mae_train=np.sum(np.abs(np.array(train_predictions) - np.array(train_labels))) / len(train_labels)
    mae_test=np.sum(np.abs(np.array(test_predictions) - np.array(test_labels))) / len(test_labels)
    
    # 绘制训练残差图##Train
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train_predictions, train_residuals, color='blue',label=f'Train MAE: {mae_train:.3f} eV')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Train Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.ylim(-1,1)
    plt.legend()
    
    # 绘制测试残差图和95%置信区间##Test
    plt.subplot(1, 2, 2)
    plt.scatter(test_predictions, test_residuals, color='green',label=f'Test MAE: {mae_test:.3f} eV')
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.axhline(y=ci_lower, color='gray', linestyle='--', label=f'95% CI Lower: {ci_lower:.2f}eV')
    plt.axhline(y=ci_upper, color='gray', linestyle='--', label=f'95% CI Upper: {ci_upper:.2f}eV')
    
    
    # 确保预测值和残差都是numpy数组
    test_predictions = np.array(test_predictions)
    test_residuals = np.array(test_residuals)
    sorted_indices = np.argsort(test_predictions)
    sorted_test_predictions = test_predictions[sorted_indices]
    sorted_test_residuals = test_residuals[sorted_indices]
    pred_min = sorted_test_predictions.min()
    pred_max = sorted_test_predictions.max()
    # 轻微扩展范围
    extension = (pred_max - pred_min) * 0.05  # 比如扩展1%
    fill_x_min = pred_min - extension
    fill_x_max = pred_max + extension
    plt.xlim(fill_x_min, fill_x_max)
    plt.ylim(-1,1)
    plt.fill_between([fill_x_min,fill_x_max], ci_lower, ci_upper, color='gray', alpha=0.3)
    
    plt.title('Test Residuals')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.legend()
    
    plt.tight_layout()
    #plt.show()
    plt.savefig('residual_12_6.png',dpi=900)
    
    
    print('Drawing R2...')
    plt.figure(figsize=(9,9))
    plt.scatter(test_labels,test_predictions, linestyle="-", marker="o", color="blue", label=f"R2:{test_r2:.4f}")
    plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], color='red', linestyle='--',)
    plt.xlabel("True Value (eV)",fontsize=14)
    plt.ylabel("Predict Value (eV)",fontsize=14)
    plt.title(f"Predictions vs True Values",fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.savefig('R2_9_9.png',dpi=900)
    
    print('Drawing predictions vs trues...')
    plt.figure(figsize=(12, 6))
    plt.plot(test_predictions, label="Predict Values", linestyle="-", marker="o", color="blue")
    plt.plot(test_labels, label="True Values", linestyle="-", marker="x", color="red")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Predictions vs True Values")
    plt.legend()
    plt.grid(True)
    plt.savefig("Pre_VS_Tru_12_6.png",dpi=900)
    print("Batch_size:",batch_size)
    print("Heads_Number:",gat_heads)
    print("Attention K neighbors:",k_hop_attention)

    def run_command(command):
        result = os.system(command)
        if result != 0:
            print(f"Command '{command}' failed with exit code {result}")
 
 
    #add './' to the front other wise it returns error code 256 which the '-' sybol would be consider as other meaning in shell
    out_dir_name=f'./{test_r2:.3f}_batch_{batch_size}_FC_4_heads_{gat_heads}_seed_{seed}_lr_{lr_rate}_opt_{opt_dis}_katt_{k_hop_attention}'
    print(f"mkdir {out_dir_name}")
    run_command(f"mkdir {out_dir_name}")
    run_command(f"mv *.png {out_dir_name}")
    run_command(f"mv model* {out_dir_name}")
    run_command(f"cp {sys.argv[0]} {out_dir_name}")
