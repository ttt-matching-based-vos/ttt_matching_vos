import torch
import torch.nn.functional as F


# Soft aggregation from STM
def aggregate(prob, keep_bg=False):
    logits = get_logits(prob)
    return get_softmax(logits, keep_bg)


def get_logits(prob):
    # add the background mask
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))
    return logits


def get_softmax(logits, keep_bg=False):
    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]


def get_log_softmax(logits, keep_bg=False):
    if keep_bg:
        return F.log_softmax(logits, dim=0)
    else:
        return F.log_softmax(logits, dim=0)[1:]


def get_entropy(logits, keep_bg=False):
    entropy = get_softmax(logits, keep_bg) * get_log_softmax(logits, keep_bg)
    entropy = -1.0 * entropy.sum()
    return entropy.mean()


def aggregate0(prob, keep_bg=False):
    # add the background mask
    new_prob = torch.cat([
        torch.prod(1-prob, dim=0, keepdim=True),
        prob
    ], 0).clamp(1e-7, 1-1e-7)
    logits = torch.log((new_prob /(1-new_prob)))

    if keep_bg:
        return F.softmax(logits, dim=0)
    else:
        return F.softmax(logits, dim=0)[1:]