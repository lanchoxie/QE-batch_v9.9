import sqlite3
import json
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep

db_file = 'gnn_data.save/wholegraphs_53d_features.db'

def get_graph_by_name(db_file, graph_name):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 根据名称查询图数据
    c.execute("SELECT graph_data FROM wholegraphs WHERE name = ?", (graph_name,))
    row = c.fetchone()

    # 关闭数据库连接
    conn.close()

    # 如果找到了图，将JSON格式的图数据转换回图结构
    if row:
        graph_data = json.loads(row[0])
        graph = nx.node_link_graph(graph_data)
        return graph
    else:
        return None

def get_all_graphs(db_file):
    # 连接到SQLite数据库
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # 查询所有图数据
    c.execute("SELECT name, graph_data FROM wholegraphs")
    rows = c.fetchall()

    # 关闭数据库连接
    conn.close()

    # 创建一个字典，用于存储名称和图结构的映射
    graphs = {}
    for name, graph_data in rows:
        graph = nx.node_link_graph(json.loads(graph_data))
        graphs[name] = graph

    return graphs

# 示例：获取名称为'name1'的图
#graph_name = 'LiNiO2-331_NCMT_6111_1'
#graph = get_graph_by_name(db_file, graph_name)
#if graph:
#    print(f"Graph {graph_name} has been retrieved from the database.")
#else:
#    print(f"Graph {graph_name} was not found in the database.")
#print(graph.nodes[0]['feature'])
## 示例：获取包含所有图的字典
#all_graphs = get_all_graphs(db_file)
#print("All graphs have been retrieved from the database.")

def extract_subgraphs_with_periodic_structure(graph, k, label_extract=None):
    # 存储已处理的li-ni边的索引对
    processed_edges = set()

    def get_neighbors_xty(node,depth):
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
    if label_extract:
        subgraphs = []
        for u, v, data in graph.edges(data=True):
            # 只考虑有能量信息的li_ni_edge边
            if data.get('edge_type') == 'li_ni_edge' and 'delta_E' in data:
                #print(u,v)
                # 无视索引对的顺序
                edge_tuple = tuple(sorted([u, v]))
    
                # 检查此索引对是否已经处理过
                if edge_tuple not in processed_edges:
                    processed_edges.add(edge_tuple)
    
                    collected_nodes_u=get_neighbors_xty(u,k)
                    collected_nodes_v=get_neighbors_xty(v,k)
                    
                    #if (u==9)and(v==37):
                        #print("U:",collected_nodes_u)
                        #print("V:",collected_nodes_v)
                    # 合并两个集合，同时去除重复元素
                    collected_nodes = collected_nodes_u.union(collected_nodes_v)
    
    
                    # 创建子图
                    subgraph = nx.Graph()
                    for node in collected_nodes:
                        subgraph.add_node(node, **graph.nodes[node])
                        subgraph.nodes[node]['original_index'] = node  # Store the original index
                        subgraph.nodes[node]['original_element'] = graph.nodes[node]['element']  # Store the original index
    
                    # 添加边，确保边的两个端点都在子图中
                    for node in collected_nodes:
                        for neighbor in graph.neighbors(node):
                            if neighbor in collected_nodes:
                                if graph.edges[node,neighbor]['edge_type']=='original':
                                    subgraph.add_edge(node, neighbor, **graph[node][neighbor])
    
                    subgraph.add_edge(u, v, **graph[u][v])
                    subgraphs.append(subgraph)

    elif not label_extract:
        subgraphs = []
        for u, v, data in graph.edges(data=True):
            # 只考虑有能量信息的li_ni_edge边
            if data.get('edge_type') == 'li_ni_edge' and 'delta_E' not in data:
                #print(u,v)
                # 无视索引对的顺序
                edge_tuple = tuple(sorted([u, v]))
    
                # 检查此索引对是否已经处理过
                if edge_tuple not in processed_edges:
                    processed_edges.add(edge_tuple)
    
                    collected_nodes_u=get_neighbors_xty(u,k)
                    collected_nodes_v=get_neighbors_xty(v,k)
                    
                    #if (u==9)and(v==37):
                        #print("U:",collected_nodes_u)
                        #print("V:",collected_nodes_v)
                    # 合并两个集合，同时去除重复元素
                    collected_nodes = collected_nodes_u.union(collected_nodes_v)
    
    
                    # 创建子图
                    subgraph = nx.Graph()
                    for node in collected_nodes:
                        subgraph.add_node(node, **graph.nodes[node])
                        subgraph.nodes[node]['original_index'] = node  # Store the original index
                        subgraph.nodes[node]['original_element'] = graph.nodes[node]['element']  # Store the original index
    
                    # 添加边，确保边的两个端点都在子图中
                    for node in collected_nodes:
                        for neighbor in graph.neighbors(node):
                            if neighbor in collected_nodes:
                                if graph.edges[node,neighbor]['edge_type']=='original':
                                    subgraph.add_edge(node, neighbor, **graph[node][neighbor])
    
                    subgraph.add_edge(u, v, **graph[u][v])
                    subgraphs.append(subgraph)
    #print(processed_edges)
    return subgraphs

def store_subgraphs_in_db(subgraphs,k,mode):
    # 连接到SQLite数据库
    conn = sqlite3.connect(f'gnn_data.save/subgraphs_k_neighbor_{k}_gnn_53d_feature_{mode}.db')
    c = conn.cursor()

    # 创建表格
    c.execute('''CREATE TABLE IF NOT EXISTS subgraphs
                 (id INTEGER PRIMARY KEY, graph_data TEXT)''')

    # 清空表中的现有记录
    c.execute("DELETE FROM subgraphs")
    # 存储每个子图
    for i, subgraph in enumerate(subgraphs):
        # 将图数据转换为JSON格式
        graph_data = json.dumps(nx.node_link_data(subgraph))
        c.execute("INSERT INTO subgraphs (graph_data) VALUES (?)", (graph_data,))

    # 提交事务并关闭连接
    conn.commit()
    conn.close()


def process_graph(G,k,data_model):
    label_ext=False if data_model=='predict' else True
    subgraphs=extract_subgraphs_with_periodic_structure(G, k,label_extract=label_ext)
    return subgraphs

def print_progress(done, total):
    percent_done = done / total * 100
    bar_length = int(percent_done / 100 * 60)
    bar = "[" + "#" * bar_length + "-" * (60 - bar_length) + "]" + f"{percent_done:.2f}%" + f"   {done}/{total}"
    print(bar, "\r", end='')

def convert_graphs(graphs,k,data_model):
    with ProcessPoolExecutor() as executor:
        # 创建future到索引的映射
        futures = {executor.submit(process_graph, graph,k,data_model): i for i, graph in enumerate(graphs)}
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
    

    return [subgraph for subgraphs in converted_graphs for subgraph in subgraphs if subgraph is not None]


all_graphs_dict = get_all_graphs(db_file)
all_graphs=[graph for name,graph in all_graphs_dict.items()]
#print(len([[u,v,data] for u,v,data in all_graphs[0].edges(data=True) if 'delta_E' in data]))
#print(len([[u,v,data] for u,v,data in all_graphs[0].edges(data=True) if 'delta_E' not in data]))

#for debugging
#graph_test = get_graph_by_name(db_file, "LiNiO2-331_NCMT_6111_8")
#print(len([[u,v,data] for u,v,data in graph_test.edges(data=True) if 'delta_E' in data]))
#print(len([[u,v,data] for u,v,data in graph_test.edges(data=True) if 'delta_E' not in data]))
#a=([print([u+1,v+1,data]) for u,v,data in graph_test.edges(data=True) if 'delta_E' in data])
print("Creating whole graphs...")
for k in range(2,6):
    print(f"\nextract subgraphs k neighbor:{k}")
    output_subgraphs_train = convert_graphs(all_graphs,k,'train') 
    print(f"\nLabeled_Number:{len(output_subgraphs_train)}")
    store_subgraphs_in_db(output_subgraphs_train,k,'train')
    output_subgraphs_predict = convert_graphs(all_graphs,k,'predict') 
    print(f"\nUnlabeled_Number:{len(output_subgraphs_predict)}")
    store_subgraphs_in_db(output_subgraphs_predict,k,'predict')
print()
