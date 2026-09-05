"""The fast path must compute the same thing as the reference path, only sooner."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from scout_fl.fl.fastpath import (auto_chunk, batched_probe, build_store, evaluate_fast,
                                  local_train_fast, probe_window, store_from_datasets)
from scout_fl.fl.models import build_model, get_flat_grad, get_flat_params, set_flat_params


def _toy(n=400, clients=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 8, 8, generator=g)
    y = torch.randint(0, 4, (n,), generator=g)
    # deliberately uneven, including clients smaller than one batch
    sizes = [8, 12, 20, 25, 30, 33, 40, 44, 48, 52, 44, 44]
    parts, off = [], 0
    for s in sizes[:clients]:
        parts.append(np.arange(off, off + s)); off += s
    return x[:off], y[:off], parts


def test_batched_probe_matches_the_reference_gradient():
    x, y, parts = _toy()
    model = build_model("small_cnn", (1, 8, 8), 4)
    store = build_store(x, y, parts, "cpu")

    g1 = torch.Generator().manual_seed(11)
    losses, grads = batched_probe(model, store, batch_size=32, chunk=5, generator=g1)
    g2 = torch.Generator().manual_seed(11)
    sel, mask = probe_window(store, 32, g2)          # the identical window

    for c in range(len(parts)):
        n = int(mask[c].sum().item())
        model.zero_grad(set_to_none=False)
        loss = F.cross_entropy(model(store.x[sel[c, :n]]), store.y[sel[c, :n]])
        loss.backward()
        ref = get_flat_grad(model).detach().numpy()
        assert np.allclose(ref, grads[c], atol=1e-5), f"gradient mismatch on client {c}"
        assert abs(float(loss.detach()) - losses[c]) < 1e-5
    model.zero_grad(set_to_none=True)


def test_short_clients_are_masked_not_padded():
    """A client holding fewer points than the batch must contribute exactly its own."""
    x, y, parts = _toy()
    store = build_store(x, y, parts, "cpu")
    sel, mask = probe_window(store, 32, torch.Generator().manual_seed(3))
    for c, p in enumerate(parts):
        n = min(len(p), 32)
        assert int(mask[c].sum().item()) == n
        real = sel[c, :n].tolist()
        assert len(set(real)) == n, "the window must be drawn without replacement"
        assert set(real) <= set(p.tolist()), "a client must only ever see its own data"


def test_probe_window_is_reproducible_from_the_generator():
    x, y, parts = _toy()
    store = build_store(x, y, parts, "cpu")
    a, _ = probe_window(store, 16, torch.Generator().manual_seed(5))
    b, _ = probe_window(store, 16, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)


def test_local_train_fast_matches_reference_sgd():
    """Same data, same order, same steps must give the same update."""
    x, y, parts = _toy()
    model = build_model("small_cnn", (1, 8, 8), 4)
    store = build_store(x, y, parts, "cpu")
    start = get_flat_params(model).clone()

    out = local_train_fast(model, store, 6, epochs=1, lr=0.05, batch_size=16,
                           generator=torch.Generator().manual_seed(9))
    set_flat_params(model, start)
    idx = store.index[6]
    order = idx[torch.randperm(int(idx.numel()), generator=torch.Generator().manual_seed(9))]
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for s in range(0, int(idx.numel()), 16):
        sel = order[s:s + 16]
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(store.x[sel]), store.y[sel]).backward()
        opt.step()
    ref = (get_flat_params(model) - start).numpy()
    assert np.allclose(ref, out["update"], atol=1e-6)
    assert out["num_samples"] == len(parts[6])


def test_max_steps_stops_the_client_early():
    x, y, parts = _toy()
    model = build_model("small_cnn", (1, 8, 8), 4)
    store = build_store(x, y, parts, "cpu")
    one = local_train_fast(model, store, 9, epochs=1, lr=0.05, batch_size=8, max_steps=1,
                           generator=torch.Generator().manual_seed(1))
    assert np.linalg.norm(one["update"]) > 0.0


def test_store_from_datasets_reconstructs_the_partition():
    from scout_fl.fl.datasets import build_client_datasets
    x, y, parts = _toy()
    cds = build_client_datasets(x, y, parts)
    store = store_from_datasets(cds, "cpu")
    assert store.counts() == [len(p) for p in parts]
    for c, p in enumerate(parts):
        assert torch.allclose(store.x[store.index[c]], x[torch.as_tensor(p)])


def test_device_cap_declines_rather_than_failing():
    x, y, parts = _toy()
    assert build_store(x, y, parts, "cpu", max_device_bytes=1.0) is None
    from scout_fl.fl.datasets import build_client_datasets
    assert store_from_datasets(build_client_datasets(x, y, parts), "cpu",
                               max_device_bytes=1.0) is None


def test_auto_chunk_is_within_bounds():
    x, y, parts = _toy()
    store = build_store(x, y, parts, "cpu")
    c = auto_chunk(store, 32, 25578, default=7)
    assert 1 <= c <= store.num_clients


def test_evaluate_fast_matches_the_server():
    from scout_fl.fl.server import FLServer
    x, y, _ = _toy()
    model = build_model("small_cnn", (1, 8, 8), 4)
    server = FLServer(model, device="cpu")
    ref_loss, ref_acc = server.evaluate(x, y)
    loss, acc = evaluate_fast(model, x, y)
    assert abs(ref_loss - loss) < 1e-5 and abs(ref_acc - acc) < 1e-9
