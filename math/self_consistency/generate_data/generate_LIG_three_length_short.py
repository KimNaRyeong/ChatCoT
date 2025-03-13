import json, os, ast
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import from_networkx

def save_LIG_image(LIG, filename="LIG_graph.png"):
    # Set up the plot
    plt.figure(figsize=(12, 12))  # Adjust the figure size as needed

    # Draw the graph
    pos = nx.spring_layout(LIG)  # Choose a layout, e.g., spring_layout
    nx.draw(LIG, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")

    # Save the graph as an image
    plt.savefig(filename, format="png")
    plt.close()

def generate_one_LIG(reasoning_paths, whole_arg_list, label, k):
    def add_weighted_edge(G, u, v, weight = 1):
        if G.has_edge(u, v):
            G[u][v]['weight'] += weight
        else:
            G.add_edge(u, v, weight = weight)
    LIG = nx.DiGraph()
    for rp in reasoning_paths:
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
        
    S_data = from_networkx(LIG)
    F_data = from_networkx(LIG)
    FA_data = from_networkx(LIG)

    S_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
    F_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)
    FA_data.edge_attr = torch.tensor([LIG[u][v]['weight'] for u, v in LIG.edges()], dtype = torch.float)

    S_node_x = []
    F_node_x = []
    FA_node_x = []

    for node in LIG.nodes():
        node = ast.literal_eval(node)
        if "answer" not in node.keys(): # function call node
            if node["function_call"] == "Calculator":
                func_vector = torch.tensor([1, 0, 0], dtype = torch.float)
            elif node["function_call"] == "Equation_solver":
                func_vector = torch.tensor([0, 1, 0], dtype = torch.float)
            else: # Do_not_use_tool
                func_vector = torch.tensor([0, 0, 1], dtype = torch.float)
            
            arg_vector = torch.zeros(20, dtype = torch.float)
            arg_list = node["arguments"]
            if arg_list != None:
                for arg in arg_list:
                    if not arg: # failed processing
                        arg_vector[-1] = 1
                    else:
                        arg_index = whole_arg_list.index(arg)
                        arg_vector[arg_index] = 1
        
        else:
            func_vector = torch.tensor([1, 1, 1], dtype = torch.float)
            arg_vector = torch.zeros(20, dtype = torch.float)
            arg_index = whole_arg_list.index(node["answer"])
            arg_vector[arg_index] = 1
        
        func_arg_vector = torch.cat((func_vector, arg_vector))

        F_node_x.append(func_vector)
        FA_node_x.append(func_arg_vector)
    
        ones_vector = torch.ones(3, dtype = torch.float)
        S_node_x.append(ones_vector)
    
    S_x_stack = np.vstack(S_node_x)
    F_x_stack = np.vstack(F_node_x)
    FA_x_stack = np.vstack(FA_node_x)

    S_data.x = torch.tensor(S_x_stack, dtype=torch.float)
    F_data.x = torch.tensor(F_x_stack, dtype=torch.float)
    FA_data.x = torch.tensor(FA_x_stack, dtype=torch.float)

    S_data.y = torch.tensor([label], dtype = torch.float)
    F_data.y = torch.tensor([label], dtype = torch.float)
    FA_data.y = torch.tensor([label], dtype = torch.float)

    return S_data, F_data, FA_data

def main():
    math_cp_reasoning_paths_file = './math_cp_reasoning_paths.json'
    math_nt_reasoning_paths_file = './math_nt_reasoning_paths.json'

    with open(math_cp_reasoning_paths_file, 'r') as rf:
        math_cp_reasoning_path_dict = json.load(rf)
    
    with open(math_nt_reasoning_paths_file, 'r') as rf:
        math_nt_reasoning_path_dict = json.load(rf)
    
    ks = math_cp_reasoning_path_dict.keys()
    # ks = ['5', '9']
    for k in ks:
        dataset_for_k_S = []
        dataset_for_k_F = []
        dataset_for_k_FA = []

        if k in math_nt_reasoning_path_dict.keys():
            # print(len(math_cp_reasoning_path_dict[k]))
            # print(len(math_nt_reasoning_path_dict[k]))
            # print(len(math_cp_reasoning_path_dict[k]+ math_nt_reasoning_path_dict[k]))
            reasoning_path_for_k = math_cp_reasoning_path_dict[k]+ math_nt_reasoning_path_dict[k]
        for data_for_one_problem in reasoning_path_for_k:
            if len(data_for_one_problem["arguments_set"]) < 20:
                S_data, F_data, FA_data = generate_one_LIG(data_for_one_problem["reasoning_paths"], data_for_one_problem["arguments_set"], data_for_one_problem["score"], k)

                dataset_for_k_S.append(S_data)
                dataset_for_k_F.append(F_data)
                dataset_for_k_FA.append(FA_data)
                
      

        print("-"*50+k+"-"*50)
        # for lim in dataset_for_k_FA:
        #     for path in lim:
        #         print(path)

        if not os.path.exists(f"/workspace/LIG_data/three_length/{k}"):
            os.makedirs(f"/workspace/LIG_data/three_length/{k}")
        
        torch.save({
            "dataset_S": dataset_for_k_S,
            "dataset_F": dataset_for_k_F,
            "dataset_FA": dataset_for_k_FA,
        }, f"/workspace/LIG_data/three_length/{k}/gcn_dataset_short.pth")
        print(f"{k}th GCN datasets saved in pth format")
        



if __name__ == "__main__":
    main()