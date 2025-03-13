import json, os
import torch

def generate_one_LIM(reasoning_paths, whole_arg_list, k):
    F_paths = []
    FA_paths = []

    for rp in reasoning_paths:
        F_path = []
        FA_path = []

        for rs in rp:
            if "function_call" in rs.keys():
                if rs["function_call"] == "Calculator":
                    func_vector = torch.tensor([1, 0], dtype = torch.float)
                elif rs["function_call"] == "Equation_solver":
                    func_vector = torch.tensor([0, 1], dtype = torch.float)
                else: # Do_not_use_tool
                    func_vector = torch.tensor([0, 0], dtype = torch.float)
                
                arg_vector = torch.zeros(20, dtype = torch.float)

                arg_list = rs["arguments"]
                if arg_list != None:
                    for arg in arg_list:
                        if not arg:
                            arg_vector[-1] = 1
                        else:
                            arg_index = whole_arg_list.index(arg)
                            arg_vector[arg_index] = 1
                    
                
            else:
                func_vector = torch.tensor([1, 1], dtype = torch.float)
                arg_vector = torch.zeros(20, dtype = torch.float)
                arg_index = whole_arg_list.index(rs["answer"])
                arg_vector[arg_index] = 1
            
            func_arg_vector = torch.cat((func_vector, arg_vector))
            F_path.append(func_vector)
            FA_path.append(func_arg_vector)
        
        while len(F_path) < int(k):
            F_path.append(torch.zeros(2, dtype=torch.float)) 
            FA_path.append(torch.zeros(22, dtype=torch.float))
        
        F_paths.append(torch.stack(F_path))
        FA_paths.append(torch.stack(FA_path))
    
    return F_paths, FA_paths


            

def main():
    math_cp_reasoning_paths_file = './math_cp_reasoning_paths.json'
    math_nt_reasoning_paths_file = './math_nt_reasoning_paths.json'

    with open(math_cp_reasoning_paths_file, 'r') as rf:
        math_cp_reasoning_path_dict = json.load(rf)
    
    with open(math_nt_reasoning_paths_file, 'r') as rf:
        math_nt_reasoning_path_dict = json.load(rf)
    
    ks = math_cp_reasoning_path_dict.keys()
    # ks = ['5']
    for k in ks:
        dataset_for_k_F = []
        dataset_for_k_FA = []
        dataset_for_k_y = []

        if k in math_nt_reasoning_path_dict.keys():
            # print(len(math_cp_reasoning_path_dict[k]))
            # print(len(math_nt_reasoning_path_dict[k]))
            # print(len(math_cp_reasoning_path_dict[k]+ math_nt_reasoning_path_dict[k]))
            reasoning_path_for_k = math_cp_reasoning_path_dict[k]+ math_nt_reasoning_path_dict[k]
        for data_for_one_problem in reasoning_path_for_k:
            if len(data_for_one_problem["arguments_set"]) < 20:
                F_paths, FA_paths = generate_one_LIM(data_for_one_problem["reasoning_paths"], data_for_one_problem["arguments_set"], k)

                y = data_for_one_problem["score"]

                dataset_for_k_F.append(torch.stack(F_paths))
                dataset_for_k_FA.append(torch.stack(FA_paths))
                dataset_for_k_y.append(y)

        
        dataset_for_k_F = torch.stack(dataset_for_k_F)
        dataset_for_k_FA = torch.stack(dataset_for_k_FA)
        dataset_for_k_y = torch.tensor(dataset_for_k_y, dtype = torch.float)

        print("-"*50+k+"-"*50)
        # for lim in dataset_for_k_FA:
        #     for path in lim:
        #         print(path)

        if not os.path.exists(f"/workspace/LIM_data/two_length/{k}"):
            os.makedirs(f"/workspace/LIM_data/two_length/{k}")
        
        torch.save({
            "dataset_F": dataset_for_k_F,
            "dataset_FA": dataset_for_k_FA,
            "y": dataset_for_k_y
        }, f"/workspace/LIM_data/two_length/{k}/lstm_dataset_short.pth")
        print(dataset_for_k_F.shape)
        print(dataset_for_k_FA.shape)
        



if __name__ == "__main__":
    main()