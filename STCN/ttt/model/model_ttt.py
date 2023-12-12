"""
model.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import torch
import torch.nn as nn

from model.network import STCN


class STCN_TTT(STCN):
    def __init__(self,):
        super().__init__(False)

    def propagate(self, frames, mask_first, mask_second, selector, k16, kf16_thin, kf16, kf8, kf4,
                  encode_first=False, mem_frame=None):
        ref_v = torch.stack([
            self.encode_value(frames[:, 0], kf16[:, 0], mask_first[:, j, 0], mask_second[:, j, 0])
            for j in range(mask_first.shape[1])
        ], 1)
        ref_k = k16[:, :, 0].unsqueeze(2)

        logits, masks = [], []
        if encode_first:
            first_logits, first_mask = self.decode(ref_v, kf16_thin[:, 0], kf8[:, 0], kf4[:, 0], selector)
            logits.append(first_logits)
            masks.append(first_mask)

        # Segment frame 1 with frame 0
        l = frames.shape[1] - 1
        mem_loop = range(1, l) if mem_frame is None else [mem_frame]
        for i in mem_loop:
            prev_logits, prev_mask = self.segment(k16[:, :, i], kf16_thin[:, i], kf8[:, i], kf4[:, i],
                                                  ref_k, ref_v, selector)

            prev_other = torch.sum(prev_mask, dim=1, keepdim=True) - prev_mask
            prev_v = torch.stack([
                self.encode_value(frames[:, i].clone(), kf16[:, i].clone(), prev_mask[:, j, None], prev_other[:, j, None])
                for j in range(prev_mask.shape[1])
            ], 1)

            ref_v = torch.cat([ref_v, prev_v], 3)
            ref_k = torch.cat([ref_k, k16[:, :, i].unsqueeze(2)], 2)

            logits.append(prev_logits)
            masks.append(prev_mask)

        # Segment frame 2 with frame 0 and 1
        last_logits, last_mask = self.segment(k16[:, :, l], kf16_thin[:, l], kf8[:, l], kf4[:, l],
                                              ref_k, ref_v, selector)
        logits.append(last_logits)
        masks.append(last_mask)

        return logits, masks

    def do_cycle_pass(self, data, backwards=True, encode_first=True):
        Fs = data['rgb']
        Ms = data['gt']
        sec_Ms = data['sec_gt']
        selector = data['selector']

        # key features never change, compute once
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs)

        # forward pass
        logits_f, masks_f = self.propagate(Fs, Ms, sec_Ms, selector, k16, kf16_thin, kf16, kf8, kf4, encode_first)

        # backward pass
        logits_b, masks_b = None, None
        if backwards:
            Ms_b = masks_f[-1][:, :, None, None]
            sec_Ms_b = torch.sum(Ms_b, dim=1, keepdim=True) - Ms_b
            logits_b, masks_b = self.propagate(Fs.flip(dims=(1,)), Ms_b, sec_Ms_b, selector,
                                               k16.flip(dims=(2,)), kf16_thin.flip(dims=(1,)),
                                               kf16.flip(dims=(1,)), kf8.flip(dims=(1,)), kf4.flip(dims=(1,)),
                                               encode_first)

        return logits_f, logits_b, masks_f, masks_b

    def do_single_pass(self, data):
        Fs = data['rgb']
        Ms = data['gt']
        sec_Ms = data['sec_gt']
        selector = data['selector']

        # key features never change, compute once
        k16, kf16_thin, kf16, kf8, kf4 = self.encode_key(Fs[:, :1])

        ref_v = torch.stack([
            self.encode_value(Fs[:, 0], kf16[:, 0], Ms[:, j, 0], sec_Ms[:, j, 0])
            for j in range(Ms.shape[1])
        ], 1)
        logits, masks = self.decode(ref_v, kf16_thin[:, 0], kf8[:, 0], kf4[:, 0], selector)

        return [logits], [masks]

    def forward(self, data):
        return self.do_cycle_pass(data)

    def copy_weights_from(self, model):
        if isinstance(model, nn.Module):
            self.load_state_dict(model.state_dict())
        else:
            self.load_state_dict(model)

    def copy_weights_to(self, model):
        model.load_state_dict(self.state_dict())

    def freeze_encoders(self):
        self.freeze_key_encoder()
        self.freeze_value_encoder()

    def freeze_decoder(self):
        for param in self.decoder.parameters():
            param.requires_grad = False

    def freeze_all_keys(self):
        self.freeze_key_encoder()
        self.freeze_key_proj()
        self.freeze_key_comp()

    def freeze_key_encoder(self):
        for param in self.key_encoder.parameters():
            param.requires_grad = False

    def freeze_value_encoder(self):
        for param in self.value_encoder.parameters():
            param.requires_grad = False

    def freeze_key_proj(self):
        for param in self.key_proj.parameters():
            param.requires_grad = False

    def freeze_key_comp(self):
        for param in self.key_comp.parameters():
            param.requires_grad = False

    def freeze_network(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_batch_norms(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

    def unfreeze_batch_norms(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(True)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(True)
                module.eval()

    def freeze_parse(self, freeze_str):
        if freeze_str is not None:
            for fm in freeze_str.lower().split(','):
                if 'enc' == fm:
                    self.freeze_encoders()
                if 'dec' == fm:
                    self.freeze_decoder()
                if 'all_keys' == fm:
                    self.freeze_all_keys()
                if 'key_enc' == fm:
                    self.freeze_key_encoder()
                if 'val_enc' == fm:
                    self.freeze_value_encoder()
                if 'key_proj' == fm:
                    self.freeze_key_proj()
                if 'key_comp' == fm:
                    self.freeze_key_comp()
                if 'net' == fm:
                    self.freeze_network()
                if 'bn' == fm:
                    self.freeze_batch_norms()
                if 'ubn' == fm:
                    self.unfreeze_batch_norms()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(name)
