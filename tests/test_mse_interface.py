"""Exercise all wrapper paths with a tiny causal model, not real checkpoints."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maple import AnchorSpec, EditConfig, Estimate, MapleEditor
from maple.backends.mse import MSEBackend
from maple.calibration import calibrate
from maple.meters import QwenReadoutMeter, estimate_loss
from maple.training import train_meter


class TinyTokenizer:
    alphabet = "_+-0123456789."

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self.alphabet.index(c) for c in text]


class TinyCausalLM(nn.Module):
    def __init__(self, chatglm=False):
        super().__init__()
        self.embedding = nn.Embedding(14, 4)
        self.rnn = nn.GRU(4, 4, batch_first=True)
        self.head = nn.Linear(4, 14)
        self.config = SimpleNamespace(model_type="tiny-interface-test")
        self.chatglm = chatglm

    def forward(self, input_ids=None, inputs_embeds=None, input_fusion=None,
                output_hidden_states=False, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        if input_fusion is not None:
            inputs_embeds = torch.cat((input_fusion, inputs_embeds[:, input_fusion.shape[1]:]), dim=1)
        state, _ = self.rnn(inputs_embeds)
        hidden = state.transpose(0, 1) if self.chatglm else state
        return SimpleNamespace(logits=self.head(state), hidden_states=(hidden,))


class TinyWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = TinyCausalLM(backbone == "chatglm3")
        self.tokenizer = TinyTokenizer()
        self.task_specific_prompt, self.language = "synthetic task", "en"
        self.backbone = backbone
        self.generate_calls = 0

    def text_embedding(self, ids):
        return self.model.embedding(ids)

    def multimodal_prompt_wrap(self, fusion):
        return fusion

    def input_processing(self, wrapped, mode):
        assert mode == "generate"
        terminal = torch.zeros(len(wrapped), 2, dtype=torch.long)
        if self.backbone == "chatglm3":
            return torch.zeros(len(wrapped), wrapped.shape[1] + 2, dtype=torch.long), None
        return torch.cat((wrapped, self.text_embedding(terminal)), dim=1), None

    def generate(self, fusion):
        self.generate_calls += 1
        prefix = self.input_processing(fusion, "generate")[0]
        if self.backbone == "chatglm3":
            prefix = torch.cat((fusion, self.text_embedding(prefix[:, fusion.shape[1]:])), dim=1)
        tokens = []
        for _ in range(4):
            ids = self.model(inputs_embeds=prefix).logits[:, -1].argmax(-1)
            tokens.append(ids)
            prefix = torch.cat((prefix, self.text_embedding(ids[:, None])), dim=1)
        results = []
        for row in torch.stack(tokens, 1):
            text = "".join(self.tokenizer.alphabet[i] for i in row)
            try:
                results.append(float(text))
            except ValueError:
                results.append(0.)
        return results


class TinyCMCM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.LLM = TinyWrapper(backbone)


def backend_inputs(backbone):
    torch.manual_seed(9)
    anchors = AnchorSpec(-1, 1, 3, positive_sign=backbone == "qwen")
    backend = MSEBackend(TinyCMCM(backbone), anchors, backbone)
    return backend, torch.randn(2, 1, 4), torch.randn(2, 3, 4)


@pytest.mark.parametrize("backbone", ["qwen", "llama2", "chatglm3"])
def test_scores_equal_sequential_teacher_forcing(backbone):
    backend, pseudo, context = backend_inputs(backbone)
    actual = backend.score(pseudo, context)
    # Independent token-at-a-time reference: no padded suffix and no EOS.
    prefix, wrapped = backend._prefix(pseudo, context)
    reference = []
    for answer in backend.answer_ids:
        sequence = prefix.clone()
        score = torch.zeros(len(pseudo))
        for token in answer:
            if backbone == "chatglm3":
                output = backend.llm.model(input_ids=sequence, input_fusion=wrapped)
            else:
                output = backend.llm.model(inputs_embeds=sequence)
            score += output.logits[:, -1].log_softmax(-1)[:, token]
            ids = torch.full((len(pseudo), 1), token, dtype=torch.long)
            extra = ids if backbone == "chatglm3" else backend.llm.text_embedding(ids)
            sequence = torch.cat((sequence, extra), dim=1)
        reference.append(score)
    assert torch.allclose(actual, torch.stack(reference, -1), atol=1e-6)


@pytest.mark.parametrize("backbone", ["qwen", "llama2", "chatglm3"])
def test_gradient_finite_difference_and_native_generation(backbone):
    backend, pseudo, context = backend_inputs(backbone)
    pseudo = pseudo[:1].clone().requires_grad_(True)
    context = context[:1]
    grad = torch.autograd.grad(backend.score_one(pseudo, context, 0).sum(), pseudo)[0]
    offset = torch.zeros_like(pseudo)
    offset[0, 0, 0] = 1e-2
    diff = (backend.score_one(pseudo.detach() + offset, context, 0)
            - backend.score_one(pseudo.detach() - offset, context, 0)) / .02
    assert torch.allclose(grad[0, 0, 0], diff.squeeze(), atol=2e-4, rtol=.03)
    geometry = calibrate(backend, [(pseudo.detach(), context)], backend.anchors, 1)
    result = MapleEditor(backend, geometry, EditConfig(radii=(0., .1))).edit(
        pseudo.detach(), context, Estimate(torch.tensor([.5])))
    assert backend.llm.generate_calls == 2
    assert result["candidates"].shape == (2, 1)
    assert all(p.grad is None and not p.requires_grad for p in backend.cmcm.parameters())


def test_qwen_readout_receives_gradient_and_does_not_modify_base():
    backend, pseudo, context = backend_inputs("qwen")
    meter = QwenReadoutMeter(4)
    base = backend.hidden(pseudo, context).clone()
    estimate_loss(meter(backend, pseudo, context), torch.tensor([-.4, .7])).backward()
    assert meter.cls_token.grad is not None and meter.cls_token.grad.abs().sum() > 0
    assert torch.equal(base, backend.hidden(pseudo, context))
    assert all(p.grad is None for p in backend.cmcm.parameters())


@pytest.mark.parametrize("kind", ["qwen_readout", "laplace"])
def test_training_writes_loadable_checkpoint(kind, tmp_path):
    backend, pseudo, context = backend_inputs("qwen" if kind == "qwen_readout" else "llama2")
    backend.encode = lambda batch: (batch["pseudo"], batch["context"])
    batch = dict(pseudo=pseudo, context=context, labels={"M": torch.tensor([-.5, .5])})
    settings = dict(kind=kind, learning_rate=.001, weight_decay=.0001,
                    epochs=2, patience=2, batch_size=2)
    result = train_meter(backend, {"train": [batch], "valid": [batch]}, settings, 4,
                         tmp_path / "meter.pt", 1111)
    payload = torch.load(tmp_path / "meter.pt", weights_only=True)
    assert payload["kind"] == kind and len(result["history"]) == 2
    assert result["best_validation_mae"] >= 0

