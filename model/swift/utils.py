import copy
import os.path
import random
import json
import logging
import torch
import numpy as np
from .kv_cache import clone_past_key_values
TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def set_logger(log_path=None):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        if log_path:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def get_cache_configuration(file_name='skip_layers.json', model_name='llama-2-13b', task_name='cnndm'):
    """
    Get the cached SWIFT configuration for LLM acceleration.
    """
    if not os.path.exists(file_name):
        print("Cache file not found.")
        return None
    with open(file_name) as f:
        data = json.load(f)
        if f'{model_name}_{task_name}' not in data.keys():
            print("Configuration not found in cache.")
            return None
        else:
            print(f"Use cached configuration in {file_name}.")
            return data[f'{model_name}_{task_name}']['attention'], data[f'{model_name}_{task_name}']['mlp']

def get_choices_list(prob_list, logits_processor=None):
    """
    Generate tree choices list based on the provided confidence.

    Parameters:
    - prob_list (list): List of probabilities.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.

    Returns:
    - list: A nested list containing choices based on the probabilities.
    """
    choices_list = []
    if logits_processor is not None:
        candidate_set = [1, 3, 5, 10]
    else:
        candidate_set = [3, 5, 8, 10]
    for idx, item in enumerate(prob_list):
        if item > 0.95:
            candidate_num = candidate_set[0]
        elif item > 0.8:
            candidate_num = candidate_set[1]
        elif item > 0.5:
            candidate_num = candidate_set[2]
        else:
            candidate_num = candidate_set[3]
        choices_list.extend([[0]*idx + [i] for i in range(candidate_num)])
    return choices_list

def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    """
    Prepare the logits processor based on the provided parameters.

    Parameters:
    - temperature (float): Softmax temperature for probability scaling.
    - repetition_penalty (float): Penalty for repeating tokens.
    - top_p (float): Nucleus sampling probability threshold.
    - top_k (int): Top-k sampling threshold.

    Returns:
    - LogitsProcessorList: A list of processors to apply to the logits.
    """
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_swift_buffers(swift_choices, device="cuda"):
    """
    Generate buffers for the swift structure based on the provided choices.

    Parameters:
    - swift_choices (list): A nested list representing tree in the swift structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the swift structure.
    """

    # Sort the swift_choices based on their lengths and then their values
    sorted_swift_choices = sorted(swift_choices, key=lambda x: (len(x), x))
    swift_len = len(sorted_swift_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_swift_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Create the attention mask for swift
    swift_attn_mask = torch.eye(swift_len, swift_len)
    swift_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            # retrieve ancestor position
            if len(cur_swift_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_swift_choice) - 1):
                ancestor_idx.append(sorted_swift_choices.index(cur_swift_choice[:c + 1]) + 1)
            swift_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the swift structure
    swift_tree_indices = torch.zeros(swift_len, dtype=torch.long)
    swift_p_indices = [0 for _ in range(swift_len - 1)]
    swift_b_indices = [[] for _ in range(swift_len - 1)]
    swift_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        b = []
        for j in range(depth_counts[i]):
            cur_swift_choice = sorted_swift_choices[start + j]
            swift_tree_indices[start + j + 1] = cur_swift_choice[-1] + TOPK * i + 1
            swift_p_indices[start + j] = 0
            if len(b) > 0:
                swift_b_indices[start + j] = copy.deepcopy(b)
            else:
                swift_b_indices[start + j] = []
            b.append(cur_swift_choice[-1] + TOPK * i + 1)
        start += depth_counts[i]

    # Generate position IDs for the swift structure
    swift_p_indices = [-1] + swift_p_indices
    swift_position_ids = torch.zeros(swift_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        swift_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for swift structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_swift_choices)):
        cur_swift_choice = sorted_swift_choices[-i - 1]
        retrieve_indice = []
        if cur_swift_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_swift_choice)):
                retrieve_indice.append(sorted_swift_choices.index(cur_swift_choice[:c + 1]))
                retrieve_paths.append(cur_swift_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    swift_p_indices = torch.tensor(swift_p_indices)
    swift_p_indices_new = swift_p_indices[retrieve_indices]
    swift_p_indices_new = swift_p_indices_new.tolist()

    swift_b_indices = [[]] + swift_b_indices
    swift_b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = swift_b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(swift_tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        swift_b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    swift_buffers = {
        "swift_attn_mask": swift_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": swift_tree_indices,
        "swift_position_ids": swift_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    swift_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in swift_buffers.items()
    }
    swift_buffers["p_indices"] = swift_p_indices_new
    swift_buffers["b_indices"] = swift_b_indices_new
    return swift_buffers


def initialize_swift(input_ids, model, max_new_tokens, past_key_values, past_key_values_data, current_length_data, logits_processor=None):
    """
    Initializes the swift structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the full model logits and swift drafting logits.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (LLM): The model containing the full LLM model.
    - max_new_tokens (int): Maximum number of tokens to generate.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.

    Returns:
    - swift_logits (torch.Tensor): Logits associated with the swift structure.
    - sample_token (torch.Tensor): Token sampled from probability distribution.
    - top1_prob (float): Probability of the sampled token.
    """
    with torch.inference_mode():
        # Pass input through the base model
        outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)
        # Obtain the logits from the full model
        if logits_processor is not None:
            last_logits = logits[:, -1]
            last_logits = logits_processor(None, last_logits)
            probabilities = torch.nn.functional.softmax(last_logits, dim=1)
            sample_token = torch.multinomial(probabilities, 1)
        else:
            sample_token = torch.argmax(logits[:, -1])
            sample_token = sample_token[None, None]
        # Pass input through the swift structure (sub-layer set)
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
    return swift_logits, sample_token, top1_prob


def swift_verify(
        model,
        input_ids=None,
        past_key_values=None,
        position_ids=None,
):
    """
    Verify the swift structure using the provided model and input.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (LLM): The model containing the full LLM model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.
    - position_ids (torch.Tensor): Positional IDs associated with the swift structure.

    Returns:
    - outputs (tuple): Contains the outputs from the model.
    - orig (torch.Tensor): Original logits from the full model.
    """
    with torch.inference_mode():
        # Pass input through the base model
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=None,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        orig = model.lm_head(outputs[0])

    return outputs, orig


def sample(logits, logits_processor, k=1):
    """
    Sample from the provided logits using the specified processor.

    Args:
    - logits (torch.Tensor): Logits to sample from.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - k (int): Number of samples to generate.

    Returns:
    - sampled_indices (torch.Tensor): Indices of the sampled tokens.
    - sampled_probs (torch.Tensor): Probabilities of the sampled tokens.
    - probabilities (torch.Tensor): Probabilities of all tokens.
    """
    logits = logits.view(-1, logits.size(-1)) # default batch size 1
    logits = logits_processor(None, logits)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, -1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
    cumulative_sum = torch.cat(
        (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

    sampled_probs = sampled_probs / (1 - cumulative_sum)
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs, probabilities


@torch.no_grad()
def swift_draft(
        model,
        input_ids=None,
        new_token_num=0,
        past_key_values_data=None,
        current_length_data=None,
        max_new_tokens=1024,
        position_ids=None,
        max_step_draft=25,
        logits_processor=None,
        stop_threshold=0.8,
):
    """
    Draft new tokens using the swift structure.

    Args:
    - model (LLM): The model containing the full LLM model.
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - new_token_num (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - max_new_tokens (int): Maximum number of tokens to generate.
    - position_ids (torch.Tensor): Positional IDs associated with the swift structure.
    - max_step_draft (int): Maximum number of draft steps to take.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - stop_threshold (float): Threshold for stopping the draft process.

    Returns:
    - draft_logits (torch.Tensor): Logits associated with the draft steps.
    - top1_prob (float): Probability of the sampled token.
    """
    draft_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        draft_past_key_values_data.append(past_key_values_data[i].clone())
    draft_current_length_data = current_length_data.clone()
    draft_past_key_values = clone_past_key_values(model, draft_past_key_values_data, draft_current_length_data)

    ss_token, ss_prob, ss_op, top1_prob = [], [], [], []
    with torch.inference_mode():
        for step_draft in range(max_step_draft):
            with model.self_draft():
                draft_outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=None,
                    past_key_values=draft_past_key_values,
                    position_ids=position_ids,
                )
            current_draft_logits = model.lm_head(draft_outputs[0])
            if logits_processor is not None:
                topk_index, topk_prob, op = sample(current_draft_logits, logits_processor, k=TOPK)
                input_ids = topk_index[:, 0].unsqueeze(0)
            else:
                top = torch.topk(current_draft_logits, TOPK, dim=-1)
                topk_index, topk_prob = top.indices, top.values
                input_ids = topk_index[:, :, 0]
                op = None
            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)
            origin_draft_probs = current_draft_logits.softmax(-1)
            argmax_prob = torch.gather(origin_draft_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
            current_threshold = argmax_prob.item()
            top1_prob.append(current_threshold)
            if current_threshold < stop_threshold or new_token_num + step_draft + 2 >= max_new_tokens:
                break
    return (torch.cat(ss_token), torch.cat(ss_prob), ss_op), top1_prob


def reset_swift_mode(
        model,
):
    """
    Resets the swift settings to their initial state.

    This function ensures that after any operations involving swift,
    the model's the swift attention mask return to their default state.
    """
    model.model.swift_mask = None


def reset_past_key_values(past_key_values):
    """
    Resets the current lengths in the past key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - past_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(past_key_values)):
        for j in range(2):
            past_key_values[i][j].current_length.fill_(0)
    return past_key_values


def generate_candidates(swift_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    """
    Generate candidates based on provided logits and indices.

    Parameters:
    - swift_logits (torch.Tensor): Logits associated with the swift structure.
    - tree_indices (list or torch.Tensor): Indices associated with a tree structure.
    - retrieve_indices (list or torch.Tensor): Indices for retrieving candidates.
    - sample_token (torch.Tensor): Token sampled from probability distribution.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.

    Returns:
    - tuple: Returns cartesian candidates and tree candidates.
    """
    sample_token = sample_token.to(tree_indices.device)

    # Greedy decoding: Select the most probable candidate from the original logits.
    candidates_logit = sample_token[0]

    # Extract the TOPK candidates from the swift logits.
    candidates_swift_logits = swift_logits[0]

    # Combine the selected candidate from the original logits with the topk swift logits.
    candidates = torch.cat([candidates_logit, candidates_swift_logits.view(-1)], dim=-1)

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[tree_indices]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = swift_logits[1]
        candidates_prob = torch.cat(
            [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
            dim=-1)

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)], dim=0)
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None

    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        swift_position_ids,
        input_ids,
        retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.

    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - swift_position_ids (torch.Tensor): Positional IDs associated with the swift structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.

    Returns:
    - tuple: Returns swift logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the swift position IDs to the length of the input sequence.
    position_ids = swift_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates.
    outputs, tree_logits = swift_verify(
        model,
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    # Reorder the obtained logits and hidden states based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]

    return logits, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - logits_processor (LogitsProcessorList): Processor to use for sampling.
    - cart_candidates_prob (torch.Tensor): Cartesian candidates probabilities.
    - op (list): List of output probabilities.
    - p_indices (list): List of parent indices.
    - tree_candidates (torch.Tensor): Tree candidates.
    - b_indices (list): List of branch indices.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]
    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        adjustflag = False
        gtp = None
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []
            for j in range(candidates.shape[0]):
                if is_eq[j]:
                    x = candidates[j, i]
                    xi = x.item()
                    if xi in candidates_set or xi == -1:
                        continue
                    candidates_set.append(xi)
                    r = random.random()
                    px = gtp[xi]
                    qx = cart_candidates_prob[j, i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = j
                        break
                    else:
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        max_id = gtp.argmax()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        if torch.equal(gtp.cpu(), torch.zeros(gtp.shape)): # multinomial error
                            gtp[max_id] = 1
                        gtp = gtp / (gtp.sum() + 1e-6)
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return best_candidate, accept_length - 1, sample_p


def get_next_point_to_probe(attn_skip_layers, mlp_skip_layers, num_hidden_layers=40):
    """
    Get the next point to probe of Bayes Optimization based on the skip layers.
    Args:
    - skip_attn_layers (list): List of indices of the skipped attention layers.
    - skip_mlp_layers (list): List of indices of the skipped MLP layers.
    - num_hidden_layers (int): Number of hidden layers in the model.

    Returns:
    - dict: A dictionary containing the next point to probe.
    """
    next_point_to_probe = {}
    for i in range(num_hidden_layers - 2):
        if (i+1) in attn_skip_layers:
            next_point_to_probe[f"x{i}"] = 1.0
        else:
            next_point_to_probe[f"x{i}"] = 0.0

    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if (i - (num_hidden_layers - 2) + 1) in mlp_skip_layers:
            next_point_to_probe[f"x{i}"] = 1.0
        else:
            next_point_to_probe[f"x{i}"] = 0.0

    return next_point_to_probe


def layer_bayes_search(optimizer, utility, num_skip_layers=34, num_hidden_layers=40):
    """
    Perform Bayesian optimization to select the next point to probe based on the utility function.
    Args:
    - optimizer (BayesianOptimization): The optimizer used for Bayesian optimization.
    - utility (function): The utility function used for Bayesian optimization.
    - num_skip_layers (int): Number of layers to skip.
    - num_hidden_layers (int): Number of hidden layers in the model.

    Returns:
    - dict: A dictionary containing the next point to probe.
    - list: List of indices of the skipped attention layers.
    - list: List of indices of the skipped MLP layers.
    """
    next_point_to_probe = optimizer.suggest(utility)
    sorted_point = sorted(next_point_to_probe.items(), reverse=True, key=lambda item: item[1])
    skip_layer_list = [k for (k, v) in sorted_point[:num_skip_layers]]
    attn_skip_layers = []
    mlp_skip_layers = []
    for i in range(num_hidden_layers - 2):
        if f"x{i}" in skip_layer_list:
            attn_skip_layers.append(i + 1)
    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if f"x{i}" in skip_layer_list:
            mlp_skip_layers.append(i - (num_hidden_layers - 2) + 1)

    return next_point_to_probe, attn_skip_layers, mlp_skip_layers


def layer_random_search(num_skip_layers=34, num_hidden_layers=40):
    """
    Randomly select layers for skipping, always keep the first and last layer
    Args:
    - num_skip_layers (int): Number of layers to skip.
    - num_hidden_layers (int): Number of hidden layers in the model.

    Returns:
    - list: List of indices of the skipped attention layers.
    - list: List of indices of the skipped MLP layers.
    """
    skip_layer_list = np.random.choice((num_hidden_layers-2) * 2, num_skip_layers, replace=False)
    attn_skip_layers = []
    mlp_skip_layers = []
    for i in range(num_hidden_layers - 2):
        if i in skip_layer_list:
            attn_skip_layers.append(i + 1)
    for i in range(num_hidden_layers - 2, (num_hidden_layers - 2) * 2):
        if i in skip_layer_list:
            mlp_skip_layers.append(i - (num_hidden_layers - 2) + 1)
    return attn_skip_layers, mlp_skip_layers


def swift_optimization(model, output_ids, input_past_key_values_data,
            input_current_length_data, new_token_num, statistics, optimizer=None, utility=None, position_ids=None):
    """
    Perform an optimization to find the optimal layer set for the model based on the draft matchness.

    Args:
    - model (LLM): The model containing the full LLM model.
    - output_ids (torch.Tensor): The output token sequences.
    - input_past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - input_current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - new_token_num (int): Counter for the new tokens added during inference.
    - statistics (dict): Dictionary containing the optimization statistics.
    - optimizer (BayesianOptimization): The optimizer used for Bayesian optimization.
    - utility (function): The utility function used for Bayesian optimization.
    - position_ids (torch.Tensor): Positional IDs associated with the swift structure.
    """
    # Avoid modifying the input_ids in-place
    generate_ids = output_ids.clone()

    cur_past_key_values_data = []
    for i in range(len(input_past_key_values_data)):
        cur_past_key_values_data.append(input_past_key_values_data[i].clone())
    cur_current_length_data = input_current_length_data.clone()
    input_past_key_values = clone_past_key_values(model, cur_past_key_values_data, cur_current_length_data)

    # preserve original layer set
    origin_attn_skip_layer_id_set, origin_mlp_skip_layer_id_set = model.get_skip_layers()

    skip_layer_num = int((model.config.num_hidden_layers - 2) * 2 * statistics["skip_ratio"])

    # select new layer set
    if (statistics["opt_iter"] + 1) % statistics["bayes_interval"] == 0 and statistics["bayes"]:
        logging.info("*" * 30 + "Bayes Search!" + "*" * 30)
        next_point_to_probe, _attn_skip_layer_id_set, _mlp_skip_layer_id_set = layer_bayes_search(
            optimizer, utility, num_skip_layers=skip_layer_num, num_hidden_layers=model.config.num_hidden_layers)
    else:
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = layer_random_search(
            num_skip_layers=skip_layer_num, num_hidden_layers=model.config.num_hidden_layers)
        next_point_to_probe = get_next_point_to_probe(_attn_skip_layer_id_set, _mlp_skip_layer_id_set, model.config.num_hidden_layers)
    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    # parallel drafting on previous decoded results
    with torch.inference_mode():
        with model.self_draft():
            step_end = statistics["context_window"] + 1
            parallel_draft_output = model.model(input_ids=generate_ids[:, :step_end],
                                                attention_mask=None,
                                                past_key_values=input_past_key_values,
                                                position_ids=position_ids)
    parallel_draft_logits = model.lm_head(parallel_draft_output[0])
    parallel_draft_output_ids = torch.argmax(parallel_draft_logits, dim=-1)
    verified_token_num = (parallel_draft_output_ids[:, :-1] == generate_ids[:, 1:step_end].to(parallel_draft_output_ids.device)).sum(-1).item()
    drafted_token_num = generate_ids[:, 1:step_end].size(-1)
    score = verified_token_num / drafted_token_num
    logging.info('opt_iter {}, matchness {:.4f}'.format(statistics["opt_iter"], score))

    # update the bayes optimizer
    optimizer.register(params=next_point_to_probe, target=score)

    if score > statistics["origin_score"]:
        logging.info("=" * 30 + 'matchness changed from {:.4f} to {:.4f}'.format(statistics["origin_score"], score) + "=" * 30)
        statistics["origin_score"] = score
        statistics["tolerance_iter"] = 0
        if score > statistics["max_score"]:
            statistics["optimization"] = False  # stop optimization
            logging.info("=" * 30 + 'Optimization Stopped because the score reaches the expected number!' + "=" * 30)
    else:
        model.set_skip_layers(origin_attn_skip_layer_id_set, origin_mlp_skip_layer_id_set)  # choose the better one
        statistics["tolerance_iter"] += 1
    statistics["opt_iter"] += 1
    if statistics["tolerance_iter"] > statistics["max_tolerance_iter"]:
        statistics["optimization"] = False  # stop optimization
        logging.info("=" * 30 + 'Optimization Stopped because the optimization iter reaches the max tolerance!' + "=" * 30)
    if statistics["opt_iter"] > statistics["max_opt_iter"]:
        statistics["optimization"] = False  # stop optimization
        logging.info("=" * 30 + 'Optimization Stopped because the optimization iter reaches the maximum!' + "=" * 30)


def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token_num,
        past_key_values_data_list,
        current_length_data,
        sample_p,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token_num (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.
    - sample_p (torch.Tensor): Probability of the sampled token.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - new_token_num (int): Updated counter for the new tokens added.
    - sample_token (torch.Tensor): Token sampled from probability distribution.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    prob = sample_p
    if logits_processor is not None:
        sample_token = torch.multinomial(prob, 1)
        sample_token = sample_token[None]
    else:
        sample_token = torch.argmax(prob)
        sample_token = sample_token[None, None]
    # Update the new token counter
    new_token_num += accept_length + 1

    return input_ids, new_token_num, sample_token