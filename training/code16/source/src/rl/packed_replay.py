"""Vectorized pointwise-policy replay, preserving the ordered PL objective."""
import torch

class PackedReplay:
    def __init__(self,rollouts,advantages,device,chunk_size=32):
        records=[(rec,adv) for r,adv in zip(rollouts,advantages) for rec in r.records if rec.included]
        self.chunks=[]
        for start in range(0,len(records),chunk_size):
            part=records[start:start+chunk_size]
            n=part[0][0].scores.numel()
            assert all(rec.scores.numel()==n for rec,adv in part)
            features=torch.cat([rec.node_features.to(device) for rec,adv in part])
            scores=torch.stack([rec.scores.to(device) for rec,adv in part])
            # Build full permutations once, not once per PPO epoch.
            orders=[];counts=[]
            for rec,adv in part:
                chosen=rec.order.detach().cpu().tolist()
                assert len(chosen)==len(set(chosen)) and 0<len(chosen)<n
                rest=sorted(set(range(n))-set(chosen))
                orders.append(chosen+rest);counts.append(len(chosen))
            permutation=torch.tensor(orders,device=device)
            valid=torch.arange(n,device=device)[None,:]<torch.tensor(counts,device=device)[:,None]
            self.chunks.append((features,scores,permutation,valid))
        self.old=torch.tensor([rec.logp_old for rec,adv in records],device=device)
        self.advantages=torch.stack([adv for rec,adv in records])

    def log_probs(self,selector,temperature):
        if selector.config.architecture!='mlp':raise ValueError('Only pointwise MLP can flatten positions')
        values=[]
        for features,scores,permutation,valid in self.chunks:
            z=selector(features,scores.flatten()).reshape_as(scores)/max(temperature,1e-6)
            ordered=z.gather(1,permutation)
            denom=torch.logcumsumexp(ordered.flip(1),dim=1).flip(1)
            values.append(((ordered-denom)*valid).sum(1))
        return torch.cat(values)
