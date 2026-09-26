"""Preserve a trained controller while widening it and adding causal history.

Duplicated hidden neurons split their outgoing weights. Zero-sum perturbations
of those outgoing weights preserve the function and allow copies to diverge in
subsequent optimization. New, older history columns start with zero weights.
"""

import torch


def _double_forward(network, normalized):
    """Check the copied float32 weights with higher-precision accumulation."""
    value = normalized.double()
    for layer in network:
        if isinstance(layer, torch.nn.Linear):
            value = torch.nn.functional.linear(value, layer.weight.double(), layer.bias.double())
        else:
            value = layer(value)
    return value


def checkpoint_hidden_sizes(checkpoint):
    state = checkpoint["model"]
    inferred = tuple(state[f"actor.network.{layer}.weight"].shape[0] for layer in (0, 2))
    if tuple(checkpoint.get("hidden_sizes", inferred)) != inferred:
        raise ValueError("Checkpoint architecture metadata disagrees with its weights")
    return inferred


def input_mapping(old_size, new_size, old_contract, new_contract, device):
    from .observations import observation_profile, valid_observation_contract
    old_history, new_history = old_contract["history"], new_contract["history"]
    old_profile, new_profile = observation_profile(old_contract), observation_profile(new_contract)
    rank = {'causal': 0, 'planar': 1, 'preview': 2}
    if 'command_feedback' in old_contract and 'command_feedback' not in new_contract:
        raise ValueError('Initialization cannot remove command feedback')
    if (not valid_observation_contract(old_contract) or not valid_observation_contract(new_contract)
            or rank[new_profile] < rank[old_profile] or new_history < old_history):
        raise ValueError("Initialization only permits extending the same causal observation contract")
    stride = old_contract["frame_size"] + 1
    new_stride = new_contract["frame_size"] + 1
    if (
        old_size - old_contract["size"] != new_size - new_contract["size"]
    ):
        raise ValueError("Initialization privileged observation contract differs")
    columns = torch.arange(stride, device=device)
    if 'command_feedback' in old_contract:
        old_core, new_core = old_contract['frame_size']-66, new_contract['frame_size']-66
        columns[old_core:old_core+66] = torch.arange(66, device=device)+new_core
    columns[-1] = new_stride - 1  # history validity follows the appended features
    causal = torch.cat([columns + i*new_stride for i in range(new_history-old_history, new_history)])
    if old_profile == 'preview':
        added = torch.arange(old_contract['size']-old_history*stride, device=device)+new_history*new_stride
        causal = torch.cat((causal, added))
    suffix = torch.arange(old_size-old_contract["size"], device=device) + new_contract["size"]
    return torch.cat((causal, suffix))


@torch.no_grad()
def _widen_network(destination, source, mapping):
    old_widths = [source[i].out_features for i in (0, 2)]
    new_widths = [destination[i].out_features for i in (0, 2)]
    if any(n < o or n % o for n, o in zip(new_widths, old_widths)):
        raise ValueError("Hidden sizes must be integer multiples of the source widths")
    groups = [torch.arange(n, device=mapping.device) % o for n, o in zip(new_widths, old_widths)]
    first, second = groups
    destination[0].weight.zero_()
    destination[0].weight[:, mapping] = source[0].weight[first]
    destination[0].bias.copy_(source[0].bias[first])
    for index, output_groups, input_groups, old_width in (
        (2, second, first, old_widths[0]),
        (4, None, second, old_widths[1]),
    ):
        weight = source[index].weight
        bias = source[index].bias
        if output_groups is not None:
            weight, bias = weight[output_groups], bias[output_groups]
        copies = len(input_groups) // old_width
        expanded = weight[:, input_groups] / copies
        if copies > 1:
            noise = torch.randn_like(expanded).reshape(len(weight), copies, old_width)
            noise -= noise.mean(1, keepdim=True)
            expanded += noise.reshape_as(expanded) * 1e-3
        destination[index].weight.copy_(expanded)
        destination[index].bias.copy_(bias)


@torch.no_grad()
def _copy_norm(destination, source, mapping, stride):
    destination.mean.zero_()
    destination.variance.fill_(1)
    destination.mean[mapping] = source.mean
    destination.variance[mapping] = source.variance
    destination.count.copy_(source.count)


@torch.no_grad()
def initialize_model(model, checkpoint, observation):
    from .learning import ActorCritic

    if checkpoint.get('action_chunk'):
        from .action_chunks import checkpoint_chunk_size
        if (getattr(model.actor,'chunk_size',1) != checkpoint_chunk_size(checkpoint)
                or observation != checkpoint['observation']):
            raise ValueError('Action chunk initialization requires the same chunk and observation contract')
        model.load_state_dict(checkpoint['model'],strict=True)
        return dict(method='exact chunk weights and normalization; fresh optimizer',
                    old_hidden_sizes=list(checkpoint_hidden_sizes(checkpoint)),
                    new_hidden_sizes=[model.actor.network[i].out_features for i in (0,2)],
                    max_output_difference={'exact_state_copy':0.})

    device = model.log_std.device
    source = ActorCritic(
        checkpoint["actor_size"], checkpoint["critic_size"], checkpoint_hidden_sizes(checkpoint)
    ).to(device)
    source.load_state_dict(checkpoint["model"], strict=True)
    differences = {}
    for name, dst_network, src_network, dst_norm, src_norm, size in (
        (
            "actor",
            model.actor.network,
            source.actor.network,
            model.actor.normalizer,
            source.actor.normalizer,
            checkpoint["actor_size"],
        ),
        (
            "critic",
            model.critic,
            source.critic,
            model.critic_norm,
            source.critic_norm,
            checkpoint["critic_size"],
        ),
    ):
        mapping = input_mapping(
            size, dst_network[0].in_features, checkpoint["observation"], observation, device
        )
        _widen_network(dst_network, src_network, mapping)
        _copy_norm(dst_norm, src_norm, mapping, observation["frame_size"] + 1)
        sample = torch.randn(64, dst_network[0].in_features, device=device)
        old = src_network(src_norm(sample[:, mapping]))
        new = dst_network(dst_norm(sample))
        differences[name] = float((new - old).abs().max())
        # TF32 may reorder low-order products when GEMM shape changes. Verify
        # transfer in full precision, independently from the training math mode.
        precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            normalized_new, normalized_old = dst_norm(sample), src_norm(sample[:, mapping])
            full_new, full_old = dst_network(normalized_new), src_network(normalized_old)
            error = float((full_new-full_old).abs().max())
            acceptable_roundoff = torch.allclose(full_new, full_old, atol=2e-5, rtol=1e-6)
            # Large critic outputs can differ by a few FP32 ULPs when ROCm
            # changes GEMM shape. Do not mistake accumulation order for changed
            # weights: retain the original absolute function-preservation gate
            # in FP64, AND a tight FP32 absolute+relative numerical gate.
            double_error = float((_double_forward(dst_network, normalized_new)
                                  - _double_forward(src_network, normalized_old)).abs().max())
        finally:
            torch.set_float32_matmul_precision(precision)
        differences[name + "_full_precision"] = error
        differences[name + "_float64_accumulation"] = double_error
        if not acceptable_roundoff or not double_error <= 2e-5:
            raise ValueError(f"{name} transfer changed the trained function: FP32={error}, FP64={double_error}")
    model.log_std.copy_(source.log_std)
    return {
        "method": "split outgoing weights for duplicated neurons; zero weights for added history and feedback",
        "old_hidden_sizes": list(checkpoint_hidden_sizes(checkpoint)),
        "new_hidden_sizes": [model.actor.network[i].out_features for i in (0, 2)],
        "old_history": checkpoint["observation"]["history"],
        "new_history": observation["history"],
        "max_output_difference": differences,
    }
