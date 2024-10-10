"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse

from fastchat.utils import str_to_torch_dtype

from evaluation_llama.eval import run_eval

from transformers import AutoTokenizer
from bayes_opt import BayesianOptimization, UtilityFunction

from model.swift.utils import *
from model.swift.modeling_llama import LlamaForCausalLM
from model.swift.kv_cache import initialize_past_key_values

def swift_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)
    swift_logits, sample_token, top1_prob = initialize_swift(input_ids, model, max_new_tokens,
                                                             past_key_values, past_key_values_data,
                                                             current_length_data, logits_processor=logits_processor)

    # Clone the prefilled past key and value states for swift optimization
    input_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        input_past_key_values_data.append(past_key_values_data[i].clone())
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    for idx in range(max_steps):
        # drafted tokens + 1 bonus verified token
        draft_token_num += len(top1_prob)
        # Initialize the swift buffer
        swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )

        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],
            input_ids,
            swift_buffers["retrieve_indices"],
        )

        best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
                swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"]
            )

        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            swift_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p
        )

        # layer set optimization
        if (new_token_num > (statistics["context_window"] + 1) and statistics["optimization"]
                and idx % statistics["opt_interval"] == 0):
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility)

        # swift drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break
    logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for swift sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of swift.",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,
        help="The interval of swift optimization.",
    )
    parser.add_argument(
        "--bayes-interval",
        type=int,
        default=25,
        help="The interval of bayesian optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-tolerance-iter",
        type=int,
        default=300,
        help="The maximum tolerance of layer set search iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of swift.",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        default=False,
        help="Layer set optimization.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Bayes Optimization of Layer set.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached SWIFT configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-num",
        type=int,
        default=10,
        help="The number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )

    args = parser.parse_args()

    args.model_name = (args.model_id + "-swift-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens)+ "-opt_interval-" + str(args.opt_interval)
                       + "-bayes_interval-" + str(args.bayes_interval) + "-max_opt-" + str(args.max_opt_iter) + "-max_tolerance-" + str(args.max_tolerance_iter)
                       + "-max_score-" + str(args.max_score) + "-context_window-" + str(args.context_window) + "-skip_ratio-" + str(args.skip_ratio))
    answer_file = f"outputs/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None

    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes=False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(model_name=args.model_id,
                                                                                  task_name=args.task_name)
    else:
        # Unified layer set initialization
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # Bayes Optimization Settings
    pbounds = {f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)} # keep the first and last layer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True)
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {"origin_score": 0, "opt_iter": 0, "tolerance_iter": 0,
                  "skip_ratio": args.skip_ratio, "acceptance_rate_list": [], "opt_interval": args.opt_interval,
                  "bayes_interval": args.bayes_interval, "max_opt_iter": args.max_opt_iter,
                  "max_tolerance_iter": args.max_tolerance_iter, "max_score": args.max_score,
                  "context_window": args.context_window, "optimization": args.optimization, "bayes": args.bayes}

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )
