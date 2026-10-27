"""One independent Python process per rollout GPU; unchanged shard RNG protocol.

CUDA IPC keeps trajectories on device. The producer retains them until the parent
has copied them onto the learner GPU and acknowledged receipt.
"""
import copy, traceback
import torch
import torch.multiprocessing as mp
from src.rl.parallel_rollouts import move
from src.rl.rollout import rollout_group

def _worker(index, backbone_config, model_config, requests, responses):
    try:
        from src.backbones.registry import build_backbone
        from src.models.set_selector import SetSelector
        torch.set_num_threads(1)
        torch.cuda.set_device(index)
        bb=build_backbone(backbone_config).to(f'cuda:{index}').eval()
        policy=SetSelector(model_config).to(f'cuda:{index}').eval()
        responses.put(('ready',None))
        while True:
            request=requests.get()
            if request is None:break
            weights,ids,mask,editable,steps,seed,kwargs=request
            policy.load_state_dict(weights)
            with torch.no_grad():
                result=rollout_group(bb,policy,ids.to(index),mask.to(index),
                    editable_positions=editable.to(index),step_counts=steps,
                    generator=torch.Generator().manual_seed(seed),**kwargs)
            responses.put(('result',result))
            assert requests.get()=='ack'
            del result,request,weights,ids,mask,editable
    except BaseException:
        responses.put(('error',traceback.format_exc()))
        raise

class ProcessRollouts:
    def __init__(self,backbone_config,selector,devices,main_backbone):
        if devices!=list(range(len(devices))):raise ValueError('Contiguous devices from zero required')
        self.devices=devices;self.backbone=main_backbone
        self.policy=copy.deepcopy(selector).to('cuda:0').eval()
        self.workers=[];ctx=mp.get_context('spawn')
        try:
            for index in devices[1:]:
                q,r=ctx.Queue(),ctx.Queue()
                p=ctx.Process(target=_worker,args=(index,backbone_config,selector.config,q,r))
                p.start();self.workers.append((p,q,r))
            for p,q,r in self.workers:
                kind,value=r.get(timeout=600)
                if kind!='ready':raise RuntimeError(value)
        except BaseException:self.close();raise

    def __call__(self,backbone,selector,input_ids,attention_mask,*,step_counts,
                 generator,editable_positions,**kwargs):
        n=len(self.devices)
        if len(step_counts)%n or len(step_counts)<n or kwargs.get('store_states',False):
            raise ValueError('Unsupported rollout group')
        seeds=torch.randint(0,2**62,(n,),generator=generator).tolist()
        width=len(step_counts)//n
        weights={k:v.detach().cpu().clone() for k,v in selector.state_dict().items()}
        ids=input_ids.cpu();mask=attention_mask.cpu();editable=editable_positions.cpu()
        for i,(p,q,r) in enumerate(self.workers,1):
            q.put((weights,ids,mask,editable,step_counts[i*width:(i+1)*width],seeds[i],kwargs))
        self.policy.load_state_dict(selector.state_dict())
        with torch.no_grad():
            result=rollout_group(self.backbone,self.policy,input_ids,attention_mask,
                editable_positions=editable_positions,step_counts=step_counts[:width],
                generator=torch.Generator().manual_seed(seeds[0]),**kwargs)
        for p,q,r in self.workers:
            kind,value=r.get(timeout=600)
            if kind!='result':raise RuntimeError(value)
            result.extend(move(value,input_ids.device))
            torch.cuda.synchronize(input_ids.device)
            del value
            q.put('ack')
        return result

    def close(self):
        for p,q,r in self.workers:
            if p.is_alive():q.put(None)
        for p,q,r in self.workers:
            p.join(timeout=10)
            if p.is_alive():p.terminate();p.join(timeout=10)
            q.close();r.close()
