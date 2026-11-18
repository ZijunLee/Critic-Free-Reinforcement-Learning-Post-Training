import torch

from cfrl.gradstats import GradStats


def test_matches_explicit_microbatch_gradients():
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 3)
    params = list(model.parameters())
    xs = [torch.randn(4, 5) for _ in range(6)]
    m = len(xs)

    # reference: per-micro-batch gradients at full-batch scale
    per_mb = []
    for x in xs:
        model.zero_grad()
        model(x).pow(2).mean().backward()
        per_mb.append(torch.cat([p.grad.reshape(-1).clone() for p in params]))
    g = torch.stack(per_mb)
    g_bar = g.mean(0)
    ref_var_mean = ((g - g_bar) ** 2).sum(1).sum() / (m - 1) / m

    # accumulation as in training: each micro-batch loss is scaled by 1/M
    model.zero_grad()
    stats = GradStats(params, m)
    for x in xs:
        (model(x).pow(2).mean() / m).backward()
        stats.update()
    out = stats.finalize()
    acc = torch.cat([p.grad.reshape(-1) for p in params])

    assert torch.allclose(acc, g_bar, atol=1e-6)
    assert abs(out["grad_sq_norm"] - float(g_bar @ g_bar)) < 1e-5
    assert abs(out["grad_var_mean"] - float(ref_var_mean)) < 1e-5


def test_skipped_microbatch_counts_as_zero_gradient():
    model = torch.nn.Linear(2, 1, bias=False)
    params = list(model.parameters())
    stats = GradStats(params, 2)
    (model(torch.ones(1, 2)).sum() / 2).backward()
    stats.update()
    stats.update()  # skipped micro-batch: no backward
    out = stats.finalize()
    # g_1 = [1, 1], g_2 = 0 -> g_bar = [.5, .5]; var_mean = (0.5 + 0.5) / 1 / 2
    assert abs(out["grad_sq_norm"] - 0.5) < 1e-6
    assert abs(out["grad_var_mean"] - 0.5) < 1e-6


def test_within_group_matches_explicit_per_sample_gradients():
    from cfrl.gradstats import WithinGroupStats

    torch.manual_seed(1)
    model = torch.nn.Linear(4, 2)
    params = list(model.parameters())
    groups = [torch.randn(3, 4) for _ in range(2)]  # 2 prompts, K = 3 samples each
    k, n_total = 3, 6

    expected = 0.0
    for x in groups:
        hs = []
        for row in x:
            model.zero_grad()
            (model(row).pow(2).sum() / n_total).backward()
            hs.append(torch.cat([p.grad.reshape(-1).clone() for p in params]))
        h = torch.stack(hs)
        expected += k * float(((h - h.mean(0)) ** 2).sum()) / (k - 1)

    model.zero_grad()
    stats = WithinGroupStats(params)
    for x in groups:
        stats.start_group()
        for row in x[:2]:
            (model(row).pow(2).sum() / n_total).backward()
            stats.add_sample()
        (model(x[2]).pow(2).sum() / n_total).backward()
        stats.add_sample()
        stats.end_group(k)
    assert abs(stats.finalize()["grad_var_within"] - expected) < 1e-6


def test_within_group_zero_samples_count_as_zero_gradient():
    from cfrl.gradstats import WithinGroupStats

    model = torch.nn.Linear(2, 1, bias=False)
    stats = WithinGroupStats(list(model.parameters()))
    model.zero_grad()
    stats.start_group()
    model(torch.ones(1, 2)).sum().backward()  # h_1 = [1, 1]
    stats.add_sample()
    stats.add_zero_samples(1)  # h_2 = 0
    stats.end_group(2)
    # mean [.5,.5]; scatter = 2 * 0.5 = 1; K * scatter / (K-1) = 2
    assert abs(stats.finalize()["grad_var_within"] - 2.0) < 1e-6
