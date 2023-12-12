import torch
import torch.nn as nn
import torch.nn.functional as F
from util.tensor_util import compute_tensor_iu

from collections import defaultdict


def get_iou_hook(values):
    return 'iou/iou', (values['hide_iou/i']+1)/(values['hide_iou/u']+1)

def get_sec_iou_hook(values):
    return 'iou/sec_iou', (values['hide_iou/sec_i']+1)/(values['hide_iou/sec_u']+1)

iou_hooks_so = [
    get_iou_hook,
]

iou_hooks_mo = [
    get_iou_hook,
    get_sec_iou_hook,
]


class EntropyLoss(nn.Module):
    def __init__(self, dim):
        super(EntropyLoss, self).__init__()
        self.dim = dim

    def forward(self, x):
        b = F.softmax(x, dim=self.dim) * F.log_softmax(x, dim=self.dim)
        b = -1.0 * b.sum(self.dim)
        return b.mean()


class ConsistencyLoss(nn.Module):
    def __init__(self, ):
        super(ConsistencyLoss, self).__init__()
        self.kl_criterion = nn.KLDivLoss(reduction="mean", log_target=True)

    def forward(self, logits):
        loss = 0.
        x = [F.log_softmax(x, 1) for x in logits]
        for i, x1 in enumerate(x):
            for j, x2 in enumerate(x):
                if i != j:
                    loss += self.kl_criterion(x1, x2)
        return loss / len(x) / (len(x)-1)


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            length = max(self.end_warm-self.start_warm, 1)
            this_p = self.top_p + (1 - self.top_p) * ((self.end_warm - it) / length)
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        print(f"raw_loss  {raw_loss.mean()} loss {loss.mean()} this_p {this_p}")
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, para, start_warm=20000, end_warm=70000, top_p=0.15):
        super().__init__()
        self.para = para
        self.bce = BootstrappedCE(start_warm, end_warm, top_p)

    def compute(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        selector = data.get('selector', None)

        for i in range(1, s):
            # Have to do it in a for-loop like this since not every entry has the second object
            # Well it's not a lot of iterations anyway
            for j in range(b):
                if selector is not None and selector[j][1] > 0.5:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1], data['cls_gt'][j:j+1,i], it)
                else:
                    loss, p = self.bce(data['logits_%d'%i][j:j+1,:2], data['cls_gt'][j:j+1,i], it)

                losses['loss_%d'%i] += loss / b
                losses['p'] += p / b / (s-1)

            losses['total_loss'] += losses['loss_%d'%i]

            new_total_i, new_total_u = compute_tensor_iu(data['mask_%d'%i]>0.5, data['gt'][:,i]>0.5)
            losses['hide_iou/i'] += new_total_i
            losses['hide_iou/u'] += new_total_u

            if selector is not None:
                new_total_i, new_total_u = compute_tensor_iu(data['sec_mask_%d'%i]>0.5, data['sec_gt'][:,i]>0.5)
                losses['hide_iou/sec_i'] += new_total_i
                losses['hide_iou/sec_u'] += new_total_u

        return losses

    def compute_auto_encoder_first_frame(self, data, it):
        losses = defaultdict(int)

        b, s, _, _, _ = data['gt'].shape
        # selector = data.get('selector', None)

        for j in range(b):
            # print(f"data['logits'][j:j+1] {data['logits'][j:j+1].shape}")
            # print(f"data['cls_gt'][j:j+1] {data['cls_gt'][j:j + 1, 0].shape}")
            loss, p = self.bce(data['logits'][j:j+1], data['cls_gt'][j:j + 1, 0], it)
            losses['loss'] += loss / b
            losses['p'] += p / b

        losses['total_loss'] += losses['loss']
        print(f"losses {losses['total_loss']}")
        losses['hide_iou/i'] = 1
        losses['hide_iou/u'] = 1
        losses['hide_iou/sec_i'] = 1
        losses['hide_iou/sec_u'] = 1
        # new_total_i, new_total_u = compute_tensor_iu(
        #     data['masks'] > 0.5, data['gt'][:, i] > 0.5)
        # losses['hide_iou/i'] += new_total_i
        # losses['hide_iou/u'] += new_total_u
        #
        # if selector is not None:
        #     new_total_i, new_total_u = compute_tensor_iu(
        #         data['sec_mask_%d' % i] > 0.5, data['sec_gt'][:, i] > 0.5)
        #     losses['hide_iou/sec_i'] += new_total_i
        #     losses['hide_iou/sec_u'] += new_total_u
        return losses