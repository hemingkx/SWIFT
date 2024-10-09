import json
import argparse
import numpy as np


def speed(jsonl_file, jsonl_file_base, datanum=10, report=True, report_sample=True):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)

    speeds=[]
    accept_lengths_list = []
    for datapoint in data[:datanum]:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        accept_lengths_list.extend(datapoint["choices"][0]['accept_lengths'])
        speeds.append(tokens/times)


    data = []
    with open(jsonl_file_base, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)
            data.append(json_obj)


    speeds0=[]
    for datapoint in data[:datanum]:
        tokens = sum(datapoint["choices"][0]['new_tokens'])
        times = sum(datapoint["choices"][0]['wall_time'])
        speeds0.append(tokens/times)

    tokens_per_second = np.array(speeds).mean()
    tokens_per_second_baseline = np.array(speeds0).mean()
    speedup_ratio = np.array(speeds).mean()/np.array(speeds0).mean()

    if report_sample:
        for i in range(datanum):
            print("Tokens per second: ", speeds[i])
            print("Tokens per second for the baseline: ", speeds0[i])
            print("Sample Speedup: {}".format(speeds[i]/speeds0[i]))
            print("Avg Speedup: {}\n".format(np.array(speeds[:i+1]).mean()/np.array(speeds0[:i+1]).mean()))

    if report:
        print("="*30, "Overall: ", "="*30)
        print("#Mean accepted tokens: ", np.mean(accept_lengths_list))
        print('Tokens per second: ', tokens_per_second)
        print('Tokens per second for the baseline: ', tokens_per_second_baseline)
        print("Speedup ratio: ", speedup_ratio)
    return tokens_per_second, tokens_per_second_baseline, speedup_ratio, accept_lengths_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    dataset = 'humaneval'
    datanum = 1000
    parser.add_argument(
        "--file-path",
        default=f'../test/{dataset}/{dataset}_{datanum}/model_answer/codellama-13b/codellama-13b-swift.jsonl',
        type=str,
        help="The file path of evaluated Speculative Decoding methods.",
    )
    parser.add_argument(
        "--base-path",
        default=f'../test/{dataset}/{dataset}_{datanum}/model_answer/codellama-13b/codellama-13b-vanilla.jsonl',
        type=str,
        help="The file path of evaluated baseline.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default='/data/models/Llama-2-13b-hf',
        type=str,
        help="The file path of evaluated baseline.",
    )

    args = parser.parse_args()

    speed(jsonl_file=args.file_path, jsonl_file_base=args.base_path, datanum=datanum)