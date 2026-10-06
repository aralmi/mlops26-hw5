"""Локальные unit-тесты критичных инвариантов; выданный check.sh не изменён."""

import copy
import unittest
from types import SimpleNamespace

import torch

from src.config import load_params
from src.train import accumulation_window, chunked_eval_loss, chunked_train_backward, lora_config, select_examples


class ToyCausal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Embedding(19, 8)
        self.lm_head = torch.nn.Linear(8, 19, bias=False)

    def get_base_model(self):
        return self


class TrainingTests(unittest.TestCase):
    def test_selection_is_reproducible_and_preserves_inputs(self):
        examples = list(range(100))
        selected, indices = select_examples(examples, 32, 42)
        self.assertEqual((selected, indices), select_examples(examples, 32, 42))
        self.assertEqual(len(selected), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(examples, list(range(100)))
        self.assertEqual(select_examples(examples, None, 42)[0], examples)
        self.assertNotEqual(indices, select_examples(examples, 32, 43)[1])
        with self.assertRaises(ValueError):
            select_examples(examples, 0, 42)

    def test_partial_accumulation_matches_separate_batch_mean(self):
        # Четыре полных микро-батча и два последних: обе группы должны
        # дать средний градиент, а последняя — отдельное обновление.
        p = torch.nn.Parameter(torch.tensor(1.0))
        gradients = []
        for i, target in enumerate([1., 2., 3., 4., 5., 6.]):
            size, end = accumulation_window(i, 6, 4)
            (p * target / size).backward()
            if end:
                gradients.append(p.grad.item())
                p.grad = None
        self.assertEqual(gradients, [2.5, 5.5])

    def test_freeze14_config_is_real(self):
        params = load_params()
        all_cfg = lora_config(params, 28, 0)
        freeze_cfg = lora_config(params, 28, 14)
        self.assertIsNone(all_cfg.layers_to_transform)
        self.assertEqual(freeze_cfg.layers_to_transform, list(range(14, 28)))
        with self.assertRaises(ValueError):
            lora_config(params, 28, 28)
        bad = copy.deepcopy(params)
        bad["lora"]["modules_to_save"] = ["lm_head"]
        with self.assertRaises(ValueError):
            lora_config(bad, 28, 0)

    def test_chunked_backward_and_evaluation_match_full_causal_loss(self):
        torch.manual_seed(7)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
        labels = torch.tensor([[-100, -100, 3, 4, 5, 6],
                               [-100, 8, 9, -100, 11, 12]])
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        a = ToyCausal()
        b = copy.deepcopy(a)
        # Toy Embedding принимает только input_ids; адаптируем сигнатуру модели.
        class Hidden(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.embedding = embedding
            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))
        a.model = Hidden(a.model)
        b.model = Hidden(b.model)
        a.lm_head.requires_grad_(False)
        b.lm_head.requires_grad_(False)
        actual = chunked_train_backward(a, batch, 2, 4)
        logits = b.lm_head(b.model(ids, None, False).last_hidden_state)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 19), labels[:, 1:].reshape(-1), ignore_index=-100,
        ) / 4
        expected.backward()
        torch.testing.assert_close(actual, expected)
        with torch.no_grad():
            torch.testing.assert_close(chunked_eval_loss(b, batch, 2), expected * 4)
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa.requires_grad:
                torch.testing.assert_close(pa.grad, pb.grad)


if __name__ == "__main__":
    unittest.main()
