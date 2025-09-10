from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class DFLSG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self,  target_batch, image_memory):
        ret = dict()
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        # target_data
        target_pid = target_batch['pids']
        target_image = target_batch['images']
        target_txt = target_batch['mlm_caption']
        image_feats, text_feats = self.base_model(target_image, target_txt)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), target_txt.argmax(dim=-1)].float()

        tri_inter_loss = objectives.compute_rbs(i_feats, t_feats, target_pid, margin=0.1, tau=0.02,
                                                logit_scale=logit_scale)
        tri_intra_loss = objectives.compute_inter(i_feats, t_feats, target_pid, margin=0.1, tau=0.02,
                                                  logit_scale=logit_scale)
        #tri_inter_loss = objectives.compute_sdm(i_feats, t_feats, target_pid,logit_scale)
        m_loss = image_memory.forward(i_feats, target_pid)
        ret.update({'mem_loss': m_loss.sum()})
        ret.update({'tri_inter_loss': tri_inter_loss})
        ret.update({'tri_intra_loss': tri_intra_loss})

        return ret

def build_model(args):
    model = DFLSG(args)
    # covert model to fp16
    convert_weights(model)
    return model
