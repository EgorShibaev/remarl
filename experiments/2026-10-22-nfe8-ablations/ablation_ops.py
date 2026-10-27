"""Explicit action supports and feature-removal controls for the NFE8 study."""
import os
import torch

MODE = os.environ.get('ABLATION_VARIANT', 'full')
TEMPERATURE = 1.0
RETENTION_BONUS = 0.0


def configure(mode, temperature=1.0, retention_bonus=0.0):
    global MODE, TEMPERATURE, RETENTION_BONUS
    assert mode in ('full','no_revision','no_hidden','no_position','bias_only','p2')
    assert temperature >= 0 and retention_bonus >= 0
    MODE, TEMPERATURE, RETENTION_BONUS = mode, temperature, retention_bonus


def pl_logprob(scores, order, temperature=1.0):
    batch, length = scores.shape
    count = order.shape[1]
    if not count:
        return scores.sum(-1)*0
    chosen = torch.zeros_like(scores, dtype=torch.bool).scatter(1,order,True)
    indices = torch.arange(length,device=scores.device).expand(batch,-1)
    rest = (indices+chosen*length).argsort(-1)[:,:length-count]
    values = (scores/temperature).gather(1,torch.cat((order,rest),-1))
    denom = torch.logcumsumexp(values.flip(-1),-1).flip(-1)
    return (values[:,:count]-denom[:,:count]).sum(-1)


def masked_pool(scores, masked):
    sizes = masked.sum(-1)
    assert (sizes == sizes[0]).all(), 'Exact monotone keep schedule required'
    batch, length = scores.shape
    indices = torch.arange(length,device=scores.device).expand(batch,-1)
    count = int(sizes[0])
    ids = (indices+(~masked)*length).argsort(-1)[:,:count]
    return ids, scores.gather(1,ids)


def select_order(scores, uniforms, count, masked):
    scores = scores + RETENTION_BONUS*(~masked).to(scores.dtype)
    noise = -torch.log(-torch.log(uniforms.clamp(1e-10,1.0)))
    logits = scores if TEMPERATURE == 0 else scores/TEMPERATURE+noise
    if MODE != 'no_revision':
        return torch.argsort(logits,descending=True,stable=(TEMPERATURE==0))[:,:count]
    masked_ids, _ = masked_pool(scores,masked)
    visible_count = scores.shape[1]-masked_ids.shape[1]
    added = count-visible_count
    assert added >= 0
    indices = torch.arange(scores.shape[1],device=scores.device).expand_as(masked)
    visible = (indices+masked*scores.shape[1]).argsort(-1)[:,:visible_count]
    local = logits.gather(1,masked_ids).argsort(-1,descending=True)[:,:added]
    return torch.cat((visible,masked_ids.gather(1,local)),-1)


def action_logprob(scores, order, masked):
    scores = scores + RETENTION_BONUS*(~masked).to(scores.dtype)
    if TEMPERATURE == 0:
        return scores.sum(-1)*0
    if MODE != 'no_revision':
        return pl_logprob(scores,order,TEMPERATURE)
    masked_ids, pool = masked_pool(scores,masked)
    visible_count = scores.shape[1]-masked_ids.shape[1]
    assert (~masked).gather(1,order[:,:visible_count]).all()
    new = order[:,visible_count:]
    assert masked.gather(1,new).all()
    inverse = torch.empty_like(scores,dtype=torch.long).scatter(
        1,masked_ids,torch.arange(pool.shape[1],device=scores.device).expand_as(masked_ids))
    return pl_logprob(pool,inverse.gather(1,new),TEMPERATURE)


def selector_class(base):
    class AblatedSelector(base):
        def __init__(self, config):
            super().__init__(config)
            self.variant = MODE
            if self.variant == 'no_hidden':
                for p in self.hidden_projection.parameters():
                    p.requires_grad_(False)
                # Projection emits zero for every possible hidden vector.
                torch.nn.init.zeros_(self.hidden_projection[1].weight)
                torch.nn.init.zeros_(self.hidden_projection[1].bias)
            if self.variant == 'no_position':
                self.position_embedding.weight.requires_grad_(False)
                torch.nn.init.zeros_(self.position_embedding.weight)
            if self.variant == 'bias_only':
                for p in self.parameters():
                    p.requires_grad_(False)
                self.position_bias = torch.nn.Parameter(torch.zeros(256))
                self.mask_bias = torch.nn.Parameter(torch.zeros(2))

        def delta(self, features):
            if self.variant == 'bias_only':
                return self.position_bias[features[:,2].long()] + self.mask_bias[features[:,1].long()]
            return super().delta(features)
    return AblatedSelector
