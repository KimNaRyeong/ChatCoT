import json, re
from sympy.parsing.latex import parse_latex
import sympy
import multiprocessing

def my_N(expr):
    result = sympy.N(expr)
    return result

def N_with_timeout(expr, timeout=60):
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(my_N, (expr,))
    try:
        result = async_result.get(timeout)
    except multiprocessing.TimeoutError:
        # print('TimeoutError')
        result = None
    return result

def my_simplify(expr):
    result = sympy.N(expr)
    return result

def simplify_with_timeout(expr, timeout=60):
    pool = multiprocessing.Pool(processes=1)
    async_result = pool.apply_async(my_simplify, (expr,))
    try:
        result = async_result.get(timeout)
    except multiprocessing.TimeoutError:
        # print('TimeoutError')
        result = None
    return result
        
def process_args_for_calculator(args):
    calculator_pattern = '\$.+?\$'
    all_equations = []
    for equ in re.findall(calculator_pattern, args):
        equ = equ.strip()
        if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
            equ = equ[1:-1]
        
        equ = equ.replace('\\%', '/ 100')

        try: 
            sympy_equ = parse_latex(equ)
            # print(f"sympy_equ: {sympy_equ}")
            # simplify_equ = simplify_with_timeout(sympy_equ)
            # print(f"symplify_equ: {simplify_equ}")
            # all_equations.append(simplify_equ)
            all_equations.append(str(sympy_equ))
        except:
            all_equations.append(False)
            # print('---')
            # print(f"Processing equations is failed: {args}")
            # equ = equ.strip()
            # if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
            #     equ = equ[1:-1]
            # equ = equ.replace('\\%', '/ 100')
            # print(f"equ: {equ}")
            # sympy_equ = parse_latex(equ)
            # print(f"sympy_equ: {sympy_equ}")
            # simplify_equ = simplify_with_timeout(sympy_equ)
            # print(f"symplify_equ: {simplify_equ}")
    
    return all_equations

def process_unknown_vars(raw_vars):
    unknown_vars = []

    # print(raw_vars)

    if '$' not in raw_vars:
        return [False]
        # pass
    # else:
        # print(raw_vars)

    for var in raw_vars.split(','):
        var = var.strip()
        if (len(var) > 2 and var[0] == '$' and var[-1] == '$'):
            var = var[1:-1]
        unknown_vars.append(var)
    
    return unknown_vars

def process_equation_system(raw_equations):
    equation_system = [var.strip() for var in raw_equations.split(',')]

    sympy_equ = []
    for equ in equation_system:
        try:
            if (len(equ) > 2 and equ[0] == '$' and equ[-1] == '$'):
                equ = equ[1:-1]
            if ('=' in equ):
                splited_equ = equ.split('=')
                equ = '{} - ({})'.format(splited_equ[0], splited_equ[-1])
            equ = equ.replace('\\%', '/ 100')
            equ = parse_latex(equ)
            # print(f"equ: {equ}")
            sympy_equ.append(str(equ))
            # print(simplify_equ)
        except:
            sympy_equ.append(False)
    
    return sympy_equ

def get_answer_boxed(content):
    pattern = '\\boxed'
    start_pos = content.rfind(pattern)
    if (start_pos == -1): return None
    answer = ''
    num_left = 0
    for i in range(start_pos + 7, len(content)):
        if (content[i] == '}' and num_left == 0):
            break
        if (content[i] == '{'):
            num_left = num_left + 1
        elif (content[i] == '}'):
            num_left = num_left - 1
        answer = answer + content[i]
    return answer

def get_reasoning_paths_and_args(reasoning_paths, k):
    processed_reasoning_paths = []
    arg_set = set()

    for i in range(5):
        rp = []
        for j, rs in enumerate(reasoning_paths[i]):

            if len(rp) >= k:
                break

            if ('\\boxed' in rs["content"]):
                answer = get_answer_boxed(rs["content"])
                rp.append({"answer": answer})
                arg_set.add(answer)
                break

            if ('the answer is' in rs["content"].lower()):
                answer = rs["content"].split('the answer is')[-1].strip()
                rp.append({"answer": answer})
                arg_set.add(answer)
                break
            

            if "which tool can we use?" in rs["content"]:
                if "calculator" in reasoning_paths[i][j+1]["content"].lower():
                    raw_args = reasoning_paths[i][j+3]["content"]
                    # print("============")
                    # print(raw_args)
                    processed_args = process_args_for_calculator(raw_args)
                    arg_set.update(processed_args)
                    rp.append({"function_call": "Calculator", "arguments": processed_args})

                    # print(processed_args)
                
                elif "equation solver" in reasoning_paths[i][j+1]["content"].lower():
                    raw_unkown_vars = reasoning_paths[i][j+3]["content"]
                    processed_unknown_vars = process_unknown_vars(raw_unkown_vars)
                    arg_set.update(processed_unknown_vars)

                    raw_equation_system = reasoning_paths[i][j+5]["content"]
                    processed_equ_system = process_equation_system(raw_equation_system)
                    arg_set.update(processed_equ_system)
                    rp.append({"function_call": "Equation_solver", "arguments": processed_unknown_vars + processed_equ_system})
                
                else:
                    rp.append({"function_call": "Do_not_use_tool", "arguments": None})
        

        
        processed_reasoning_paths.append(rp)
    return processed_reasoning_paths, arg_set
        
        
                              



            
            



def main():
    result_file = "./result/math_nt/chatcot_w_sc/turbo-w_sc-5shot_fixed.json"
    # result_file = "./result/math_cp/chatcot_w_sc/test.json"

    with open(result_file, 'r') as rf:
        content = json.load(rf)

    reasoning_paths_dict = dict()
    ks = range(1, 10) # max_hop = 8 + answer
    # ks = [2]
    # ks = [9]

    max_arg_len = 0
    for k in ks:
        reasoning_paths_dict[k] = []
        for problem_and_answer in content:
            # print("---")
            # print(problem_and_answer["problem"])
            reasoning_paths = problem_and_answer["chat_and_reason"]
            processed_reasoning_paths, arg_set = get_reasoning_paths_and_args(reasoning_paths, k)

            if False in arg_set:
                arg_set.remove(False)
                arg_list = list(arg_set)
                arg_list.append(False)
            else:
                arg_list = list(arg_set)

            arg_len = len(arg_list)
            if arg_len > max_arg_len:
                max_arg_len = arg_len
            
            for rp in processed_reasoning_paths:
                if 'answer' in rp[-1].keys() and 'answer' == None:
                    print(problem_and_answer['problem'])
                    
            score = problem_and_answer["score"]

            reasoning_paths_dict[k].append({"problem": problem_and_answer["problem"], "reasoning_paths": processed_reasoning_paths, "arguments_set": arg_list, "score": score})
    
    print(f"Max argument length: {max_arg_len}")

            
    
    reasoning_paths_file = './math_nt_reasoning_paths.json'

    with open(reasoning_paths_file, 'w') as wf:
        json.dump(reasoning_paths_dict, wf, indent=4)
    




if __name__ == "__main__":
    main()