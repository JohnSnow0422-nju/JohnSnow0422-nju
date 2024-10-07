import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = '/Users/johnsnow/Desktop/科技政策/2019共现矩阵.xlsx'
df = pd.read_excel(file_path, index_col=0)

# 创建空的NetworkX图
G = nx.Graph()

# 添加节点
for node in df.index:
    G.add_node(node)

# 添加边
for i in range(len(df.index)):
    for j in range(i+1, len(df.columns)):
        if df.iat[i, j] > 0:  # 只添加权重大于0的边
            G.add_edge(df.index[i], df.columns[j], weight=df.iat[i, j])

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 定义节点颜色
def rgb_to_tuple(rgb_str):
    rgb_str = rgb_str.strip('rgb()')
    r, g, b = map(int, rgb_str.split(','))
    return (r/255, g/255, b/255)

color_map = {
    'Government': rgb_to_tuple('rgb(0,0,255)'),  # Dark Blue
    'Enterprise': rgb_to_tuple('rgb(100,149,237))'),  # Blue
    'Citizen': rgb_to_tuple('rgb(0,191,255)'),   # Deep Sky Blue
    'Social Organization': rgb_to_tuple('rgb(176,224,230)'),     # Light Sky Blue
}
default_color = rgb_to_tuple('rgb(211,211,211)')  # Light Gray
node_colors = [color_map.get(node, default_color) for node in G.nodes()]

# 绘制网络图
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G)
weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
nx.draw_networkx_edges(G, pos, edgelist=weights.keys(), edge_color=rgb_to_tuple('rgb(0,191,255)'), width=[v * 0.002 for v in weights.values()])
nx.draw_networkx_labels(G, pos, font_size=8)

plt.title('2019', fontsize=8)
plt.show()
