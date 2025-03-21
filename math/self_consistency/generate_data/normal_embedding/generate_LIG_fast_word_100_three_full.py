import json, os, ast
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx
import fasttext
import fasttext.util
# fasttext pretrained 모델 로드

fasttext.util.download_model('en', if_exists='ignore')
model = fasttext.load_model('cc.en.300.bin')

embedding_size = 100
if embedding_size != 300:
    fasttext.util.reduce_model(model, embedding_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_LIG_image(LIG, filename="LIG_graph.png"):
    # Set up the plot
    plt.figure(figsize=(12, 12))  # Adjust the figure size as needed

    # Draw the graph
    pos = nx.spring_layout(LIG)  # Choose a layout, e.g., spring_layout
    nx.draw(LIG, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")

    # Save the graph as an image
    plt.savefig(filename, format="png")
    plt.close()

def embed_reasoning_path(reasoning_steps, word_vector):
    vectors = []
    for rs in reasoning_steps:
        if word_vector:
            vectors.append(model.get_word_vector(rs))
        else:
            vectors.append(model.get_sentence_vector(rs))
    
    vectors = np.stack(vectors)
    vectors_tensor = torch.from_numpy(vectors)

    return vectors_tensor

def generate_one_LIG(reasoning_paths, whole_arg_list, label, k, embedding_type = 'F'):
    def add_weighted_edge(G, u, v, weight = 1):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight = weight)

    LIG = nx.DiGraph()
    for rp in reasoning_paths:
        if embedding_type == 'F':
            if (not LIG.has_node(str(rp[0]))) and 'answer' not in rp[0].keys():
                LIG.add_node(str(rp[0]))
            for i, rs in enumerate(rp[1:]):
                if i+1 < int(k) and 'answer' not in rs.keys():
                    if not LIG.has_node(str(rs)):
                        LIG.add_node(str(rs))
                    add_weighted_edge(LIG, str(rp[i]), str(rs))
        elif embedding_type == 'FA':
            if not LIG.has_node(str(rp[0])):
                LIG.add_node(str(rp[0]))
            for i, rs in enumerate(rp[1:]):
                if i+1 < int(k):
                    if not LIG.has_node(str(rs)):
                        LIG.add_node(str(rs))
                    add_weighted_edge(LIG, str(rp[i]), str(rs))
    
    # try:
    #     save_LIG_image(LIG, filename=f"./images/LIG.png")
    # except:
    #     pass
    if len(LIG.nodes()) == 0:
        LIG.add_node('0')
    
    data = from_networkx(LIG)
    data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype=torch.float)
    node_text = []
    for node in LIG.nodes():
        node_dict = ast.literal_eval(node)
        if node_dict == 0:
            node_text.append('0')
            break
        if embedding_type == 'F':
            node_text.append(node_dict['function_call'])
        elif embedding_type == 'FA':
            if 'answer' in node_dict.keys():
                node_text.append(node_dict['answer'])
            else:
                node_text.append(f"{node_dict['function_call']}({node_dict['arguments']})")
    
    node_embedding = embed_reasoning_path(node_text, True)
    data.x = node_embedding
    data.y = torch.tensor([label], dtype=torch.float)

    return data



def main():
    math_cp_reasoning_paths_file = './math_cp_reasoning_paths.json'
    math_nt_reasoning_paths_file = './math_nt_reasoning_paths.json'

    with open(math_cp_reasoning_paths_file, 'r') as rf:
        math_cp_reasoning_path_dict = json.load(rf)
    
    with open(math_nt_reasoning_paths_file, 'r') as rf:
        math_nt_reasoning_path_dict = json.load(rf)

    ks_for_f = range(1, 9)
    ks_for_fa = range(1, 10)

    math_reasoning_paths = math_cp_reasoning_path_dict['9'] + math_nt_reasoning_path_dict['9']
    for k in ks_for_f:
        dataset_for_k_F = []
        for data in math_reasoning_paths:
            F_data = generate_one_LIG(data["reasoning_paths"], data["arguments_set"], data["score"], k, 'F')
            dataset_for_k_F.append(F_data)
        
        if not os.path.exists(f"/workspace/LIG_data/normal_embedding/three_length/{k}"):
            os.makedirs(f"/workspace/LIG_data/normal_embedding/three_length/{k}")
        
        torch.save({
            "dataset_F": dataset_for_k_F,
        }, f"/workspace/LIG_data/normal_embedding/three_length/{k}/gcn_f_dataset_full.pth")
    
    for k in ks_for_fa:
        dataset_for_k_FA = []
        for data in math_reasoning_paths:
            FA_data = generate_one_LIG(data["reasoning_paths"], data["arguments_set"], data["score"], k, 'FA')
            dataset_for_k_FA.append(FA_data)
        
        if not os.path.exists(f"/workspace/LIG_data/normal_embedding/three_length/{k}"):
            os.makedirs(f"/workspace/LIG_data/normal_embedding/three_length/{k}")
        
        torch.save({
            "dataset_FA": dataset_for_k_FA,
        }, f"/workspace/LIG_data/normal_embedding/three_length/{k}/gcn_fa_dataset_full.pth")
        

if __name__ == "__main__":
    main()