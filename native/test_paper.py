from pathlib import Path
import pytest
from paper_protocol import official, prompt, score
from batch_grpo_recipe import Recipe
from src.decoding.schedules import topk_schedule


@pytest.fixture
def api():
    return official(Path(__file__).parent / "official")


@pytest.mark.parametrize("task,text,gold,expected", [
    ("gsm8k", r"Reasoning. \boxed{42}", 42., 1),
    ("gsm8k", "<answer>42</answer>", 42., 1),
    ("gsm8k", "42", 42., 0),
    ("gsm8k", r"\boxed{43}", 42., 0),
    ("gsm8k", r"\boxed{-5}", None, 0),
    ("math", r"\boxed{\frac{1}{2}}", r"\frac{1}{2}", 1),
    ("math", r"\boxed{0.5}", r"\frac{1}{2}", 1),
    ("math", r"\boxed{\frac{1}{3}}", r"\frac{1}{2}", 0),
    ("math", r"\boxed{1} then \boxed{2}", "2", 1),
    ("math", r"\boxed{}", "", 1),
])
def test_official_scorer(api, task, text, gold, expected):
    assert score(api, text, dict(task=task, answer=gold))["reward"] == expected


def test_prompt_exact_task_constants(api):
    class Tokenizer:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt and not tokenize
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return messages[0]["content"]
    for task, constant in (("gsm8k", "GSM_SYSTEM_PROMPT"), ("math", "MATH500_SYSTEM_PROMPT")):
        assert prompt(api, Tokenizer(), "Q", task) == api[constant] + "\n\nQ<reasoning>"
    assert api["MATH500_SYSTEM_PROMPT"].endswith('</answer>"\n')
    assert api["extract_answer_gsm8k"](None, "#### -5") is None


@pytest.mark.parametrize("nfe", [4, 6, 8, 16, 32, 64])
def test_recipe_and_nondivisor_nfe(nfe):
    r = Recipe(training_questions=14973, groups_per_minibatch=16, passes=4,
               target_accepted_groups=65536, lr_schedule="constant", learning_rate=1e-4, nfe=nfe)
    assert r.total_batches == 4096 and r.updates_per_batch == 4
    assert r.total_batches * r.updates_per_batch == 16384
    assert r.learning_rate_at(4095) == 1e-4
    keep = [topk_schedule("linear", s, nfe, 256) for s in range(1, nfe+1)]
    assert keep[-1] == 256 and keep == sorted(set(keep))
    if nfe == 6:
        assert keep == [43, 86, 128, 171, 214, 256]
