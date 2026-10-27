"""Synchronous rollout shards; one central policy update and global G advantage.

Each worker holds an independent frozen backbone and a freshly synchronized policy.
Independent shard RNG seeds are drawn from the checkpointed CPU action generator.
No cross-prompt batching, stale actors, local advantage normalization, or DDP loss.
"""
import copy
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import torch
from src.rl.rollout import rollout_group


def move(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device)
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name:move(getattr(value,f.name),device)
                                            for f in dataclasses.fields(value)})
    if isinstance(value,list):return [move(v,device) for v in value]
    return value


class ParallelRollouts:
    def __init__(self, backbones, selector, devices, *, parallel=True):
        if len(backbones)!=len(devices) or not devices:
            raise ValueError('One frozen backbone required per device')
        self.devices=[torch.device(d) for d in devices]
        self.backbones=backbones
        self.selectors=[copy.deepcopy(selector).to(d).eval() for d in self.devices]
        self.parallel=parallel
        self.executor=ThreadPoolExecutor(max_workers=len(devices))

    def close(self):self.executor.shutdown(wait=True)

    def __call__(self, backbone, selector, input_ids, attention_mask, *, step_counts,
                 generator, editable_positions, **kwargs):
        n=len(self.devices)
        if len(step_counts)%n or len(step_counts)<n:
            raise ValueError('Group must divide equally across workers')
        if kwargs.get('store_states',False):
            raise ValueError('Distributed collector supports frozen backbone only')
        seeds=torch.randint(0,2**62,(n,),generator=generator).tolist()
        for replica in self.selectors:replica.load_state_dict(selector.state_dict())
        width=len(step_counts)//n
        def work(i):
            device=self.devices[i]
            context=torch.cuda.device(device) if device.type=='cuda' else nullcontext()
            with context,torch.no_grad():
                result=rollout_group(self.backbones[i],self.selectors[i],input_ids.to(device),
                    attention_mask.to(device),editable_positions=editable_positions.to(device),
                    step_counts=step_counts[i*width:(i+1)*width],
                    generator=torch.Generator().manual_seed(seeds[i]),**kwargs)
                # Compact detached trajectories only; no vocabulary-logit transfer.
                return move(result,input_ids.device)
        if self.parallel:
            futures=[self.executor.submit(work,i) for i in range(n)]
            shards=[f.result() for f in futures]
        else:shards=[work(i) for i in range(n)]
        return [r for shard in shards for r in shard]
