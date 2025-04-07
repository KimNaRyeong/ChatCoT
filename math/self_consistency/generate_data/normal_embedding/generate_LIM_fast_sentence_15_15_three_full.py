import json, os
import torch
import numpy as np
import fasttext
import fasttext.util
from itertools import zip_longest

fasttext.util.download_model('en', if_exists='ignore')
model_for_fc = fasttext.load_model('cc.en.300.bin')
model_for_answer = fasttext.load_model('cc.en.300.bin')

embedding_size = 30
if embedding_size != 300:
    fasttext.util.reduce_model(model_for_answer, embedding_size)
    fasttext.util.reduce_model(model_for_fc, embedding_size // 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_one_trans_embedding(reasoning_paths, embedding_type = 'F'):
    # F_paths = []
    # FA_paths = []
    if embedding_type == 'FA':
        max_seq_len = 9
    elif embedding_type == 'F':
        max_seq_len = 8
    # reasoning_paths_for_embedding = []
    # for rp in reasoning_paths:
    #     rp_for_embedding = []
    #     for rs in rp:
    #         if "function_call" in rs.keys():
    #             if embedding_type == 'F':
    #                 rp_for_embedding.append(rs["function_call"])
    #             elif embedding_type == 'FA':
    #                 rp_for_embedding.append(f"{rs['function_call']}({rs['arguments']})")
    #         else:
    #             if embedding_type == 'FA':
    #                 rp_for_embedding.append(rs["answer"])
    #     reasoning_paths_for_embedding.append(rp_for_embedding)

    trans_reasoning_paths = list(zip_longest(*reasoning_paths, fillvalue = '0'))

    trans_reasoning_paths += [['0']*5]*(max_seq_len-len(trans_reasoning_paths))

    trans_rps_embedding = []
    for reasoning_steps in trans_reasoning_paths:
        embed_reasoning_steps = []
        for rs in reasoning_steps:
            if "function_call" in rs:
                func_emb = embed_reasoning_path(str(rs["function_call"]), False, True)
                arg_emb = embed_reasoning_path(str(rs["arguments"]), False, True)
                fc_emb =  np.concatenate((func_emb, arg_emb), axis = 0)
                embed_reasoning_steps.append(fc_emb)
            else:
                if "answer" in rs:
                    answer_emb = embed_reasoning_path(str(rs["answer"]), False, False)
                else:
                    answer_emb = embed_reasoning_path(rs, False, False)
                embed_reasoning_steps.append(answer_emb)

        reasoning_steps_vectors = np.stack(embed_reasoning_steps)
        rs_embedding = torch.from_numpy(reasoning_steps_vectors)
        trans_rps_embedding.append(rs_embedding)
    return trans_rps_embedding
    

    # for rp in reasoning_paths:
    #     F_path = []
    #     FA_path = []

    #     for rs in rp:
    #         if "function_call" in rs.keys():
    #             if rs["function_call"] == "Calculator":
    #                 func_vector = torch.tensor([1, 0, 0], dtype = torch.float)
    #             elif rs["function_call"] == "Equation_solver":
    #                 func_vector = torch.tensor([0, 1, 0], dtype = torch.float)
    #             else: # Do_not_use_tool
    #                 func_vector = torch.tensor([0, 0, 1], dtype = torch.float)
                
    #             arg_vector = torch.zeros(48, dtype = torch.float)

    #             arg_list = rs["arguments"]
    #             if arg_list != None:
    #                 for arg in arg_list:
    #                     if not arg:
    #                         arg_vector[-1] = 1
    #                     else:
    #                         arg_index = whole_arg_list.index(arg)
    #                         arg_vector[arg_index] = 1
                    
                
    #         else:
    #             func_vector = torch.tensor([1, 1, 1], dtype = torch.float)
    #             arg_vector = torch.zeros(48, dtype = torch.float)
    #             arg_index = whole_arg_list.index(rs["answer"])
    #             arg_vector[arg_index] = 1
            
    #         func_arg_vector = torch.cat((func_vector, arg_vector))
    #         F_path.append(func_vector)
    #         FA_path.append(func_arg_vector)
        
    #     while len(F_path) < int(k):
    #         F_path.append(torch.zeros(3, dtype=torch.float)) 
    #         FA_path.append(torch.zeros(51, dtype=torch.float))
        
    #     F_paths.append(torch.stack(F_path))
    #     FA_paths.append(torch.stack(FA_path))
    
    # return F_paths, FA_paths

def embed_reasoning_path(rs, word_vector, is_fc):
    vectors = []
    if is_fc:
        model = model_for_fc
    else:
        model = model_for_answer

    if word_vector:
        return model.get_word_vector(rs)
    else:
        print(rs)
        return model.get_sentence_vector(rs)
    # for rs in reasoning_steps:
    #     rs = rs.replace("\n", " ").strip()
    #     if word_vector:
    #         vectors.append(model.get_word_vector(rs))
    #     else:
    #         vectors.append(model.get_sentence_vector(rs))
    
    # vectors = np.stack(vectors)
    # vectors_tensor = torch.from_numpy(vectors)

    return vectors_tensor


            

def main():
    math_cp_reasoning_paths_file = './math_cp_reasoning_paths.json'
    math_nt_reasoning_paths_file = './math_nt_reasoning_paths.json'

    with open(math_cp_reasoning_paths_file, 'r') as rf:
        math_cp_reasoning_path_dict = json.load(rf)
    
    with open(math_nt_reasoning_paths_file, 'r') as rf:
        math_nt_reasoning_path_dict = json.load(rf)
    
    ks_for_fa = range(1, 10)
    math_reasoning_paths = math_cp_reasoning_path_dict['9'] + math_nt_reasoning_path_dict['9']

    FA_trans_embeddings = []
    ys = []

    for data in math_reasoning_paths:
        FA_trans_embeddings.append(generate_one_trans_embedding(data["reasoning_paths"], 'FA'))
        ys.append(data["score"])


    
    for k in ks_for_fa:
        dataset_for_k_FA = []

        for trans_emb in FA_trans_embeddings:
            trans_emb_until_k = trans_emb[:int(k)]
            rps_emb = torch.stack(trans_emb_until_k, dim=1)
            dataset_for_k_FA.append(rps_emb)
        
        dataset_for_k_FA = torch.stack(dataset_for_k_FA)
        dataset_for_k_y = torch.tensor(ys, dtype=torch.float)

        # print("-"*50+k+"-"*50)
        # for lim in dataset_for_k_FA:
        #     for path in lim:
        #         print(path)

        if not os.path.exists(f"/workspace/LIM_data/normal_embedding/three_length/concat/{k}"):
            os.makedirs(f"/workspace/LIM_data/normal_embedding/three_length/concat/{k}")
        
        torch.save({
            "dataset_FA": dataset_for_k_FA,
            "y": dataset_for_k_y
        }, f"/workspace/LIM_data/normal_embedding/three_length/concat/{k}/lstm_fa_dataset_full.pth")
        print(dataset_for_k_FA.shape)
        



if __name__ == "__main__":
    main()