# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from collections import OrderedDict

import config

PHI_MAPPING = {
    "transformer.embd.wte.weight": "model.embed_tokens.weight",
    "lm_head.linear": "lm_head",
    "final_layernorm": "norm",
    "transformer": "model",
    ".h.": ".layers.",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "dense": "o_proj",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.fc2": "mlp.down_proj",
}


def map_key(origin_key):
    for k, v in PHI_MAPPING.items():
        if k in origin_key:
            origin_key = origin_key.replace(k, v)
    return origin_key


def find_weight_by_subname(v_dict, subname):
    value_list = [value for key, value in v_dict.items() if subname in key]
    if len(value_list) != 1:
        raise ValueError("Found too many or too few matches in state dict")
    return value_list[0]


# def convert_phi_weights():
#     new_dict = OrderedDict()
#     original_weights_keys = sorted(config.state_dict.keys())

#     for original_weights_key in original_weights_keys:
#         new_key = map_key(original_weights_key)
#         new_dict[new_key] = config.state_dict[original_weights_key]

#     return new_dict

def convert_phi_weights():
    converted_weights = OrderedDict()
    original_weights_keys = sorted(config.state_dict.keys())

    for original_weights_key in original_weights_keys:
        new_key = original_weights_key

        if "rotary_emb" in new_key:
            continue

        if "qkv_proj" in new_key:
            if "weight" in new_key:
                weight = config.state_dict[new_key]
                weights_shape = weight.shape
                weight = (
                    weight.view(3, config.num_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                ).view(config.num_heads, 3, -1)
                q_proj_key = map_key(new_key.replace("qkv_proj", "q_proj"))
                k_proj_key = map_key(new_key.replace("qkv_proj", "k_proj"))
                v_proj_key = map_key(new_key.replace("qkv_proj", "v_proj"))
                converted_weights[q_proj_key] = weight[:, 0, ...].reshape(config.hidden_size, config.hidden_size)
                converted_weights[k_proj_key] = weight[:, 1, ...].reshape(config.hidden_size, config.hidden_size)
                converted_weights[v_proj_key] = weight[:, 2, ...].reshape(config.hidden_size, config.hidden_size)

                config.state_dict.pop(new_key)

            elif "bias" in new_key:
                bias = config.state_dict[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_heads, -1).transpose(0, 1).reshape(*bias_shape).view(config.num_heads, 3, -1)
                q_proj_key = map_key(new_key.replace("qkv_proj", "q_proj"))
                k_proj_key = map_key(new_key.replace("qkv_proj", "k_proj"))
                v_proj_key = map_key(new_key.replace("qkv_proj", "v_proj"))
                converted_weights[q_proj_key] = bias[..., 0, :].reshape(config.hidden_size)
                converted_weights[k_proj_key] = bias[..., 1, :].reshape(config.hidden_size)
                converted_weights[v_proj_key] = bias[..., 2, :].reshape(config.hidden_size)

                config.state_dict.pop(new_key)

            continue

        new_key = map_key(new_key)

        converted_weights[new_key] = config.state_dict.pop(original_weights_key)

    return converted_weights
