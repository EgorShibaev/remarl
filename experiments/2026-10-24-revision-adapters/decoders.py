"""Prespecified revision reevaluations, with actual decoder-call accounting."""
from pathlib import Path
from adapters import remdm_uncode, wino


def decode(model, prompt, cfg):
    if cfg['mode']=='wino':
        return wino(model,prompt,source=Path(__file__).parent/'sources/WINO/LLaDA/decoding.py',
                    block=cfg['block'],threshold=cfg['threshold'],threshold_back=cfg['threshold_back'])
    if cfg['mode']=='remdm_uncode':
        return remdm_uncode(model,prompt,k=cfg['k'],loops=cfg['loops'],initial_fraction=.875)
    raise ValueError(cfg)


def grid(kind):
    configs=[dict(mode='wino',block=b,threshold=.6,threshold_back=.9,seed=101)
             for b in (256,128)]
    # Half the calls generate the canvas; half are two-call revision loops.
    # The shortest compatible 7/8-prefill preset is 16 calls. No fabricated
    # four/eight-call point or uncounted finalization is added.
    for steps in (16,32,64,128,256):
        configs.append(dict(mode='remdm_uncode',block=256,steps=steps,k=512//steps,
                            loops=steps//4,seed=101))
    splits=[('test_gsm8k',1319),('test_math',500)] if kind=='math' else [('test_humaneval',164),('test_mbpp',500)]
    cells=[]
    for cfg in configs:
        setting=f"nfe{cfg['steps']}-k{cfg['k']}-loops{cfg['loops']}" if 'steps' in cfg else 'th0.6-back0.9'
        for split,count in splits:
            cells.append(dict(**cfg,split=split,count=count,
                              name=f"{cfg['mode']}-bl{cfg['block']}-{setting}-{split}-s101"))
    return cells
