import json
import matplotlib
import matplotlib.pyplot as plt

def main():
    math_cp_reasoning_paths_file = "math/self_consistency/math_cp_reasoning_paths.json"
    math_nt_reasoning_paths_file = "math/self_consistency/math_nt_reasoning_paths.json"

    with open(math_cp_reasoning_paths_file, 'r') as rf:
        math_cp_reasoning_paths = json.load(rf)
    with open(math_nt_reasoning_paths_file, 'r') as rf:
        math_nt_reasoning_paths = json.load(rf)
    
    reasoning_paths_dict = math_cp_reasoning_paths['9'] + math_nt_reasoning_paths['9']

    length_dict = {i: 0 for i in range(1, 10)}
    
    for problem_and_rps in reasoning_paths_dict:
        for rp in problem_and_rps["reasoning_paths"]:
            length_dict[len(rp)] += 1
    
    print(length_dict)

    # draw a graph
    x = range(1, 10)
    y = [length_dict[i] for i in x]

    plt.bar(x, y, color='skyblue')
    plt.xlabel('Length')
    plt.xlabel('Number of reasoning paths')
    plt.title('Number of reasoning paths with length x')
    plt.xticks(x)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # matplotlib.use('TkAgg')
    plt.savefig("math/self_consistency/length_analysis.png")



if __name__ == "__main__":
    main()
