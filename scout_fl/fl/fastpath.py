"""Device resident data and a batched probe, which is where the run time actually goes.

Profiling one round at the campaign operating point (N=100, budget 10, CIFAR-10,
small CNN) puts 69 percent of the time in the probe, 25 percent in local training and
6 percent in evaluation. The probe is 100 separate forward and backward passes on one
mini batch each, so it is launch bound rather than compute bound. Each of those passes
also rebuilt a DataLoader, copied a batch from host to device, and pulled a flat
gradient back, which on CUDA means a synchronisation per client per round.

This module removes all three costs.

* ``ClientTensorStore`` holds the whole training subsample on the device once and
  addresses each client by an index tensor, so no batch is ever copied again and no
  DataLoader is constructed inside the round loop.
* ``batched_probe`` computes every client's probe gradient in one vectorised call
  through ``torch.func``, in chunks that bound the activation memory. One hundred
  launches become four, and one device to host transfer replaces one hundred.
* ``local_train_fast`` trains a selected client by indexing a permutation instead of
  iterating a DataLoader, and accumulates its loss on the device so the only
  synchronisation is at the end of the client.

The probe is exact rather than approximate. Clients holding fewer samples than the
batch size are padded and masked, so the masked mean reproduces the gradient the
unbatched path computes on the same sample, and the padding contributes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

_HAS_FUNC = True
try:  # torch.func landed in 2.0; fall back cleanly on older builds
    from torch.func import functional_call, grad_and_value, vmap
except Exception:  # pragma: no cover - exercised only on old torch
    _HAS_FUNC = False


@dataclass
class ClientTensorStore:
    """The training subsample, resident on the device, plus per-client index tensors."""

    x: torch.Tensor                 # (N, ...) on device
    y: torch.Tensor                 # (N,) on device
    index: list[torch.Tensor]       # per client, indices into x/y, on device
    device: str
    pad_index: torch.Tensor | None = None   # (C, maxn) client indices, right padded
    lengths: torch.Tensor | None = None     # (C,) true client sizes
    _valid: torch.Tensor | None = None      # (C, maxn) bool, True where the entry is real

    def plan(self) -> None:
        """Precompute the padded index matrix once, since the partition never changes.

        With it, drawing one probe window for every client is a single argsort and a
        single gather rather than a Python loop over clients.
        """
        if self.pad_index is not None:
            return
        dev = self.x.device
        lens = torch.tensor([int(i.numel()) for i in self.index], device=dev)
        maxn = int(lens.max().item()) if lens.numel() else 1
        pad = torch.zeros((len(self.index), maxn), dtype=torch.long, device=dev)
        valid = torch.zeros((len(self.index), maxn), dtype=torch.bool, device=dev)
        for c, idx in enumerate(self.index):
            n = int(idx.numel())
            if n:
                pad[c, :n] = idx
                pad[c, n:] = idx[0]          # padding repeats a real row, then gets masked
                valid[c, :n] = True
        self.pad_index, self.lengths, self._valid = pad, lens, valid

    @property
    def num_clients(self) -> int:
        return len(self.index)

    def counts(self) -> list[int]:
        return [int(i.numel()) for i in self.index]

    def nbytes(self) -> int:
        return self.x.element_size() * self.x.nelement() + self.y.element_size() * self.y.nelement()


def build_store(x: torch.Tensor, y: torch.Tensor, parts, device: str,
                max_device_bytes: float | None = None) -> ClientTensorStore | None:
    """Move the training subsample to the device once, or return None if it will not fit.

    ``max_device_bytes`` is a cap on what the feature tensor may occupy. Returning None
    lets the caller keep the original host resident path rather than risk an allocation
    failure, which matters when several campaign shards share one card.
    """
    need = x.element_size() * x.nelement() + y.element_size() * y.nelement()
    if max_device_bytes is not None and need > float(max_device_bytes):
        return None
    try:
        xd = x.to(device, non_blocking=True)
        yd = y.to(device, non_blocking=True)
    except (RuntimeError, MemoryError):
        return None
    idx = [torch.as_tensor(np.asarray(p), dtype=torch.long, device=device) for p in parts]
    return ClientTensorStore(xd, yd, idx, str(device))


def store_from_datasets(client_datasets, device: str,
                        max_device_bytes: float | None = None) -> ClientTensorStore | None:
    """Build a device resident store from the per-client TensorDatasets.

    Each sample belongs to exactly one client, so concatenating the client tensors
    reconstructs the training subsample without duplicating it, and the client index is
    a contiguous range. Returns None when the data will not fit under the cap, which
    leaves the caller on the original host resident path.
    """
    if not client_datasets:
        return None
    try:
        xs = [d.tensors[0] for d in client_datasets]
        ys = [d.tensors[1] for d in client_datasets]
    except AttributeError:
        return None
    n = sum(int(t.shape[0]) for t in xs)
    need = xs[0].element_size() * n * int(np.prod(xs[0].shape[1:])) + 8 * n
    if max_device_bytes is not None and need > float(max_device_bytes):
        return None
    try:
        x = torch.cat(xs).to(device, non_blocking=True)
        y = torch.cat(ys).to(device, non_blocking=True)
    except (RuntimeError, MemoryError):
        return None
    idx, off = [], 0
    for t in xs:
        k = int(t.shape[0])
        idx.append(torch.arange(off, off + k, dtype=torch.long, device=device))
        off += k
    return ClientTensorStore(x, y, idx, str(device))


def _flat_grad_like(model: torch.nn.Module, grads: dict[str, torch.Tensor],
                    lead: int) -> torch.Tensor:
    """Flatten a vmapped per-client gradient dict in ``model.parameters()`` order."""
    return torch.cat([grads[name].reshape(lead, -1) for name, _ in model.named_parameters()], dim=1)


def probe_window(store: "ClientTensorStore", batch_size: int,
                 generator: torch.Generator | None = None):
    """Draw one fixed-size probe window per client, without replacement, in one pass.

    Sorting uniform noise with the padded positions pushed to the end gives each row a
    random permutation prefix of that client's own samples. Clients holding fewer than
    ``batch_size`` points keep padding in the tail, and the returned mask removes it from
    both the loss and the gradient. Returns (indices (C, B), mask (C, B)).
    """
    store.plan()
    device = store.x.device
    C, maxn = store.pad_index.shape
    noise = torch.rand((C, maxn), device=device, generator=generator)
    noise = noise.masked_fill(~store._valid, 2.0)
    order = noise.argsort(dim=1)[:, :batch_size]
    sel = store.pad_index.gather(1, order)
    take = store.lengths.clamp(max=batch_size).unsqueeze(1)
    mask = (torch.arange(batch_size, device=device).unsqueeze(0) < take).to(store.x.dtype)
    return sel, mask


def auto_chunk(store: ClientTensorStore, batch_size: int, num_params: int,
               default: int = 25, share: float = 1.0) -> int:
    """Pick how many clients to differentiate together from the free device memory.

    The activation footprint grows linearly in the chunk, so the chunk follows from a
    conservative estimate of the bytes one client needs and a budget of a quarter of the
    free memory. ``share`` is this process's slice of the card, which matters because a
    campaign shard sees the whole device as free while several shards are running, and
    each of them would otherwise size itself as though it had the card to itself. On any
    backend without a memory query the default stands.
    """
    from scout_fl.utils.device import free_memory_bytes

    free = free_memory_bytes(str(store.x.device))
    if free is None:
        return max(int(default), 1)
    per_sample = int(np.prod(store.x.shape[1:]))
    # forward activations, their gradients, and the per-client parameter gradient
    per_client = 4 * batch_size * per_sample * 24 + 4 * num_params
    budget = 0.25 * float(free) * max(float(share), 1e-3)
    return int(max(1, min(store.num_clients, budget // max(per_client, 1))))


def batched_probe(model: torch.nn.Module, store: ClientTensorStore, *,
                  batch_size: int = 64, chunk: int = 25,
                  generator: torch.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Probe every client on the current global model. Returns (losses, gradients).

    ``gradients`` is (num_clients, num_params) and ``losses`` is (num_clients,), both as
    NumPy arrays, transferred once. ``chunk`` bounds how many clients are differentiated
    together and therefore bounds the activation memory. On an allocation failure the
    chunk is halved and the attempt repeated, down to a single client, so a crowded card
    degrades in speed rather than failing the run.
    """
    if not _HAS_FUNC:
        raise RuntimeError("torch.func is unavailable; use the host-resident probe")
    device = store.x.device
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_fn(p, xb, yb, mask):
        out = functional_call(model, (p, buffers), (xb,))
        per = F.cross_entropy(out, yb, reduction="none")
        return (per * mask).sum() / mask.sum().clamp(min=1.0)

    per_client = vmap(grad_and_value(loss_fn), in_dims=(None, 0, 0, 0))

    sel, M = probe_window(store, batch_size, generator)
    X, Y = store.x[sel], store.y[sel]                         # one gather each

    grads_out, loss_out = [], []
    i, step = 0, max(int(chunk), 1)
    while i < X.shape[0]:
        j = min(i + step, X.shape[0])
        try:
            g, l = per_client(params, X[i:j], Y[i:j], M[i:j])
        except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError,
                MemoryError):
            if step == 1:
                raise
            step = max(step // 2, 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        grads_out.append(_flat_grad_like(model, g, j - i))
        loss_out.append(l)
        i = j
    grads = torch.cat(grads_out).cpu().numpy()
    losses = torch.cat(loss_out).cpu().numpy()
    return losses, grads


def local_train_fast(model: torch.nn.Module, store: ClientTensorStore, client: int, *,
                     epochs: int = 1, lr: float = 0.05, batch_size: int = 64,
                     optimizer: str = "sgd", momentum: float = 0.9,
                     max_steps: int | None = None,
                     generator: torch.Generator | None = None) -> dict:
    """Local SGD for one client, indexing device resident tensors.

    ``model`` must already carry the global parameters. The loss accumulates on the
    device, so the only host synchronisation is the single transfer of the update.
    """
    from scout_fl.fl.models import get_flat_params

    idx = store.index[client]
    n = int(idx.numel())
    init = get_flat_params(model).clone()
    model.train()
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    device = store.x.device
    total = torch.zeros((), device=device)
    seen, steps, done = 0, 0, False
    for _ in range(epochs):
        order = idx[torch.randperm(n, device=device, generator=generator)]
        for s in range(0, n, batch_size):
            sel = order[s:s + batch_size]
            xb, yb = store.x[sel], store.y[sel]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach() * sel.numel()
            seen += int(sel.numel())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True
                break
        if done:
            break
    update = (get_flat_params(model) - init).cpu().numpy()
    return {"update": update, "loss": float(total.item()) / max(seen, 1), "num_samples": n}


def evaluate_fast(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
                  batch_size: int = 1024) -> tuple[float, float]:
    """Test loss and accuracy with one synchronisation instead of two per batch."""
    model.eval()
    device = next(model.parameters()).device
    if x.device != device:
        x, y = x.to(device), y.to(device)
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    with torch.no_grad():
        for i in range(0, y.numel(), batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            out = model(xb)
            loss_sum += F.cross_entropy(out, yb, reduction="sum")
            correct += (out.argmax(1) == yb).sum()
    n = int(y.numel())
    return float(loss_sum.item()) / n, float(correct.item()) / n
