"""Frozen Dream decoder with its native one-position prediction shift."""

from pathlib import Path
import torch

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
DREAM_REVISION = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"


class DreamAnswerModel:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)

    @classmethod
    def from_local_checkpoint(cls, path, device):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(Path(path)),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device)
        return cls(model)

    @torch.no_grad()
    def __call__(self, ids, attention, positions):
        # Dream's official generation shifts logits right: prediction for i uses
        # decoder output i-1. The policy reads that same output's hidden state.
        previous = positions - 1
        if (previous < 0).any():
            raise ValueError("Dream requires a prompt token before the answer")
        token_positions = attention.long().cumsum(-1) - 1
        token_positions.masked_fill_(attention == 0, 1)
        valid = attention.bool()
        full_attention = valid[:, None, :, None] & valid[:, None, None, :]
        out = self.model.model(
            input_ids=ids,
            attention_mask=full_attention,
            position_ids=token_positions,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state.gather(
            1, previous[..., None].expand(-1, -1, out.last_hidden_state.shape[-1])
        )
        return self.model.lm_head(hidden), hidden
