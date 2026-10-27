"""Generate one answer with a released ReMaRL checkpoint; never execute code."""
import argparse
from pathlib import Path
import sys
import json
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'native'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--task',choices=['gsm8k','math','raw'],default='gsm8k')
    p.add_argument('--nfe',type=int,required=True)
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--backend',choices=['torch','vllm'],default='torch')
    args=p.parse_args()
    if not 1<=args.nfe<=256:p.error('--nfe must be between 1 and 256')
    import torch
    from transformers import AutoTokenizer
    from model import LLaDAOfflineModel
    from src.models.set_selector import SetSelector,SetSelectorConfig
    from worker import collect
    from batch_grpo_core import state_digest
    from paper_protocol import official,prompt
    saved=torch.load(args.policy,map_location='cpu',weights_only=True)
    if state_digest(saved['model'])!=saved['policy_sha256']:
        raise ValueError('Policy-state hash mismatch')
    cfg=SetSelectorConfig(**saved['selector_config'])
    policy=SetSelector(cfg).to(args.device).eval()
    policy.load_state_dict(saved['model'],strict=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    formatted=args.prompt if args.task=='raw' else prompt(official(ROOT/'native/official'),tokenizer,args.prompt,args.task)
    prefix=tokenizer(formatted,truncation=False)['input_ids']
    length=cfg.position_count
    ids=torch.tensor([prefix+[126336]*length],dtype=torch.long,device=args.device)
    positions=torch.arange(len(prefix),len(prefix)+length,device=args.device)[None,:]
    engine=LLaDAOfflineModel.from_local_checkpoint(args.model,device=args.device,backend=args.backend,fuse_projections=True)
    if ids.shape[1]>engine.config.max_sequence_length:
        p.error('Prompt exceeds the backbone context; this wrapper does not silently truncate or extend it')
    noise=torch.Generator().manual_seed(args.seed)
    uniforms=torch.stack([torch.rand(1,length,generator=noise) for _ in range(args.nfe)])
    result=collect(SimpleNamespace(mask_token_id=126336),policy,ids,torch.ones_like(ids),positions,
                   uniforms,answer_backbone=engine,record=False)
    answer=result.final_ids.gather(1,positions)[0].tolist()
    print(json.dumps(dict(text=tokenizer.decode(answer,skip_special_tokens=True),answer_token_ids=answer,
                         nfe=args.nfe,seed=args.seed,policy_sha256=saved['policy_sha256']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
