from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from translation.manager import Manager


def triu_mask(size: int, device: str | None = None) -> Tensor:
    mask = torch.ones((1, size, size), device=device)
    return torch.triu(mask, diagonal=1) == 0


def greedy_search(manager: 'Manager', src_encs: Tensor, max_length: int = 512) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    tgt_mask = model.rnn_pool(tgt_mask, mask=True)
    tgt_mask = model.rnn_pool(tgt_mask.transpose(1, 2), mask=True).transpose(1, 2)
    path = torch.full((1, max_length), vocab.BOS, device=device)

    N = manager.kernel_size
    for i in range(N, max_length, N):
        tgt_mask_ = tgt_mask[:, : i // N, : i // N]
        tgt_encs = model.decode(src_encs, path[:, :i], tgt_mask=tgt_mask_)
        h = tgt_encs[:, -1].unsqueeze(1)
        x = torch.zeros((h.size(0), 1, h.size(2)), device=h.device)
        for j in range(N):
            h, _ = model.rnn_dec(x, h)
            logits = model.out_embed(h.unsqueeze(1), inverse=True)
            path[0, i + j] = logits.log_softmax(dim=-1).argmax(dim=-1)
            if path[0, i + j] == vocab.EOS:
                return path[0, : i + j].squeeze(0)
            x = model.tgt_embed(path[0:1, i + j].unsqueeze(0))
        # print(vocab.denumberize(path[0, : i + N].tolist()))

    return path.squeeze(0)


def beam_search(
    manager: 'Manager', src_encs: Tensor, beam_size: int = 4, max_length: int = 512
) -> Tensor:
    model, vocab, device = manager.model, manager.vocab, manager.device
    tgt_mask = triu_mask(max_length, device=device)
    active = torch.ones(beam_size, dtype=torch.bool, device=device)
    paths = torch.full((beam_size, max_length), vocab.BOS, device=device)
    probs = torch.zeros(beam_size, device=device)

    i, init_size = 0, beam_size
    while (i := i + 1) < max_length and beam_size > 0:
        tgt_encs = model.decode(
            src_encs.expand(beam_size, -1, -1), paths[active, :i], tgt_mask=tgt_mask[:, :i, :i]
        )
        logits = model.out_embed(tgt_encs[:, -1], inverse=True)
        scores = probs[active].unsqueeze(1) + logits.log_softmax(dim=-1)
        if i == 1:
            scores = scores[0]

        topv, topi = torch.topk(scores.flatten(), beam_size)
        if beam_size < init_size:
            active[~active] |= probs[~active] < topv.max() / i
            active_count = int(active.count_nonzero())
            if active_count > beam_size:
                beam_size = active_count
                topv, topi = torch.topk(scores.flatten(), beam_size)

        reorder = topi // vocab.size()
        paths[active] = paths[active][reorder]
        paths[active, i] = topi % vocab.size()
        probs[active] = topv

        terminated = paths[:, i] == vocab.EOS
        probs[terminated] /= i
        active &= ~terminated
        beam_size = int(active.count_nonzero())

    return paths[probs.argmax()]
