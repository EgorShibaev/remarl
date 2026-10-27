from __future__ import annotations

import torch


def js_divergence_from_logits(
    base_logits: torch.Tensor,
    cf_logits: torch.Tensor,
    topk: int | None = None,
) -> torch.Tensor:
    """Jensen-Shannon divergence for matching rows of token logits."""
    if topk is not None and topk > 0 and topk < base_logits.shape[-1]:
        union = torch.unique(
            torch.cat(
                [
                    torch.topk(base_logits, topk, dim=-1).indices.flatten(),
                    torch.topk(cf_logits, topk, dim=-1).indices.flatten(),
                ],
                dim=0,
            )
        )
        base_logits = base_logits.index_select(-1, union)
        cf_logits = cf_logits.index_select(-1, union)
    p = torch.softmax(base_logits, dim=-1)
    q = torch.softmax(cf_logits, dim=-1)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=-1)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def true_token_logprob_change(
    base_logits: torch.Tensor,
    cf_logits: torch.Tensor,
    true_token_ids: torch.Tensor,
) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    cf_log_probs = torch.log_softmax(cf_logits, dim=-1)
    row = torch.arange(true_token_ids.numel(), device=true_token_ids.device)
    return (base_log_probs[row, true_token_ids] - cf_log_probs[row, true_token_ids]).abs()


def top1_flip(base_logits: torch.Tensor, cf_logits: torch.Tensor) -> torch.Tensor:
    return base_logits.argmax(dim=-1).ne(cf_logits.argmax(dim=-1)).float()


def soft_conflict_label(score: torch.Tensor, tau: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((score - tau) / max(temperature, 1e-6))

