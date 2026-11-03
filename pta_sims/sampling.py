"""
Running Prometheus' NUTS sampler with a watchdog on the warm-up.

Why this exists
---------------
On a 4000-dimensional PTA posterior, NUTS occasionally warms up into a region
it cannot integrate: dual averaging drives the step size toward zero, and
because a smaller step needs a longer trajectory to cover the same distance,
the tree depth saturates at the same time.  Each iteration then costs
``2^max_tree_depth`` gradient evaluations and the chain goes nowhere.  This is a
failure of the sampler, not of the model -- a different random seed usually
warms up fine -- but left alone it burns hours and yields a chain that looks
superficially complete.

``numpyro.infer.MCMC`` gives no way to intervene: warm-up runs to completion
before any state is visible.  So this module drives NUTS through numpyro's
functional ``hmc`` interface in a plain Python loop, which costs nothing at
these iteration times (tens of milliseconds each, dominated by the gradient)
and makes every iteration's step size, tree depth and acceptance probability
available as it happens.  When the collapse signature appears during warm-up
the attempt is abandoned immediately and restarted from a new seed.

The watchdog only ever fires during warm-up.  After adaptation is frozen, a
small step size is a property of the posterior, not a pathology, and killing
the run would bias the result.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


@dataclass
class AttemptRecord:
    """What happened on one attempt at a chain."""

    seed: int
    outcome: str                       # 'ok', 'collapsed', 'diverged', 'timeout'
    iterations: int
    elapsed: float
    final_step_size: float
    mean_accept_prob: float
    reason: str = ''


@dataclass
class SamplerDiagnostics:
    """Health of the chain that was finally kept."""

    seed: int
    num_warmup: int
    num_samples: int
    elapsed: float
    step_size: float
    mean_accept_prob: float
    divergences: int
    max_depth_fraction: float
    warmup_step_size: np.ndarray = field(default_factory=lambda: np.empty(0))
    warmup_accept_prob: np.ndarray = field(default_factory=lambda: np.empty(0))
    warmup_tree_depth: np.ndarray = field(default_factory=lambda: np.empty(0))
    sample_tree_depth_mean: float = 0.0
    attempts: list = field(default_factory=list)

    def summary(self):
        return {
            'seed': int(self.seed),
            'elapsed_s': float(self.elapsed),
            'step_size': float(self.step_size),
            'mean_accept_prob': float(self.mean_accept_prob),
            'divergences': int(self.divergences),
            'max_depth_fraction': float(self.max_depth_fraction),
            'mean_tree_depth': float(self.sample_tree_depth_mean),
            'n_attempts': len(self.attempts),
            'attempts': [vars(a) for a in self.attempts],
        }


def _canonical_floats(values):
    """
    Put every starting value on JAX's current default float dtype.

    The starting point is usually assembled by hand -- an injected truth, a
    python float, a float32 array left over from somewhere else -- so its
    dtypes need not agree with each other.  They used to agree by accident,
    because JAX ran in x64-off mode and cast everything to float32 on the way
    in.  Prometheus now enables x64 globally, so a mixed dict survives intact
    and NUTS fails deep inside its tree-doubling loop, where ``lax.cond``
    insists both branches return identical types and the error names neither
    the site nor the caller.  One cast up front costs nothing and keeps that
    from happening.
    """
    import jax.numpy as jnp

    canonical = jnp.result_type(float)
    out = {}
    for name, value in values.items():
        array = jnp.asarray(value)
        out[name] = (array.astype(canonical)
                     if jnp.issubdtype(array.dtype, jnp.floating) else array)
    return out


def unconstrained_init(model, values, seed=0, model_args=(), model_kwargs=None):
    """
    Map constrained parameter values to the unconstrained space NUTS works in.

    Use this to start a chain at the injected truth.  Sites absent from
    ``values`` are initialized by numpyro's default strategy.

    Parameters
    ----------
    model : callable
        A numpyro model, e.g. ``pta_model.sampling_model``.
    values : dict
        Constrained values keyed by site name.
    seed : int
    model_args, model_kwargs

    Returns
    -------
    (init_params, potential_fn, transforms) : tuple
        ``init_params`` is the unconstrained dict to hand to the sampler,
        ``potential_fn`` the negative log density on that space, and
        ``transforms`` maps each site back to constrained space.
    """
    import jax.random as jr
    from numpyro.distributions import biject_to
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    info = initialize_model(
        jr.PRNGKey(seed), model,
        init_strategy=init_to_value(values=_canonical_floats(values)),
        model_args=model_args, model_kwargs=model_kwargs or {})

    transforms = {}
    for name, site in info.model_trace.items():
        if site['type'] != 'sample' or site.get('is_observed', False):
            continue
        support = getattr(site['fn'], 'support', None)
        if support is not None:
            transforms[name] = biject_to(support)

    return info.param_info.z, info.potential_fn, transforms


def run_nuts_monitored(model, init_values, num_warmup=1000, num_samples=5000,
                       seed=0, *, collect=None, max_tree_depth=10,
                       target_accept_prob=0.8, dense_mass=False,
                       step_size_floor=1e-5, check_after=150, check_window=50,
                       saturation_fraction=0.9, max_attempts=4,
                       jitter_scale=0.05, report_every=100,
                       wall_clock_limit=None, progress=None,
                       checkpoint=None, checkpoint_every=5000,
                       model_args=(), model_kwargs=None, use_scan=False):
    """
    Run NUTS from a given starting point, restarting if warm-up collapses.

    Parameters
    ----------
    model : callable
        numpyro model, e.g. ``pta_model.sampling_model``.
    init_values : dict
        Constrained starting values -- the injected truth, for an
        injection/recovery study.  Sites left out get numpyro's default init.
    num_warmup, num_samples : int
    seed : int
        Base seed.  Attempt ``k`` uses ``seed + 10007 * k``, so a restart
        changes both the sampler stream and (through ``jitter_scale``) the
        starting point.
    collect : sequence of str, optional
        Which sites to keep.  Defaults to everything except the Fourier
        coefficient site ``z``, which is 4020 numbers per sample and is almost
        never what you want to store.
    max_tree_depth, target_accept_prob, dense_mass
        Passed through to NUTS.
    step_size_floor : float
        Step sizes below this count as collapsed.  1e-5 is the value the
        Prometheus examples quote; a healthy chain on the 15 yr array settles
        near 1e-2.
    check_after : int
        Do not judge before this warm-up iteration.  Early adaptation swings
        the step size around by orders of magnitude and is not a failure.
    check_window : int
        Number of recent iterations the verdict is based on.
    saturation_fraction : float
        Fraction of the window that must hit ``max_tree_depth`` for the
        collapse verdict.  Requiring *both* signals matters: a small step size
        alone can be a genuinely difficult posterior, and a saturated tree
        alone can be a long but productive trajectory.  Together they mean the
        sampler is taking the maximum number of steps and still not moving.
    max_attempts : int
        How many seeds to try before giving up.
    jitter_scale : float
        Standard deviation of the unconstrained-space perturbation added to the
        starting point on attempts after the first.  Restarting from exactly
        the same point with only a new sampler stream often re-collapses.
    report_every : int
        Iterations between ``progress`` callbacks.  Also caps the chunk length
        the sampler runs inside a single compiled ``lax.scan``.
    wall_clock_limit : float, optional
        Seconds per attempt before giving up on it.
    progress : callable, optional
        Called with a dict of live diagnostics: ``phase`` ('warmup'/'sample'),
        ``iteration``, ``total``, ``step_size``, ``accept_prob``,
        ``tree_depth``, ``divergences``, ``elapsed``, ``attempt``.
    checkpoint : callable, optional
        Called every ``checkpoint_every`` samples (and at the end) with
        ``(samples_so_far, info)``: the constrained samples collected so far
        as a dict of arrays, and the same dict ``progress`` receives.  Use it
        to save a long run incrementally.
    checkpoint_every : int

    Returns
    -------
    (samples, diagnostics) : (dict of ndarray, SamplerDiagnostics)

    Raises
    ------
    RuntimeError
        If every attempt collapsed.
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from numpyro.infer.hmc import hmc

    attempts = []
    for attempt in range(max_attempts):
        attempt_seed = seed + 10007 * attempt
        init_params, potential_fn, transforms = unconstrained_init(
            model, init_values, seed=attempt_seed,
            model_args=model_args, model_kwargs=model_kwargs)

        if attempt > 0 and jitter_scale > 0:
            key = jr.PRNGKey(attempt_seed)
            jittered = {}
            for i, (name, value) in enumerate(sorted(init_params.items())):
                jittered[name] = value + jitter_scale * jr.normal(
                    jr.fold_in(key, i), value.shape)
            init_params = jittered

        keep = list(collect) if collect is not None else [
            k for k in init_params if k != 'z']

        result = _run_one_attempt(
            potential_fn, init_params, transforms, keep,
            num_warmup=num_warmup, num_samples=num_samples,
            seed=attempt_seed, attempt=attempt,
            max_tree_depth=max_tree_depth,
            target_accept_prob=target_accept_prob, dense_mass=dense_mass,
            step_size_floor=step_size_floor, check_after=check_after,
            check_window=check_window, saturation_fraction=saturation_fraction,
            report_every=report_every, wall_clock_limit=wall_clock_limit,
            progress=progress, checkpoint=checkpoint,
            checkpoint_every=checkpoint_every, hmc=hmc, jax=jax, jnp=jnp, jr=jr,
            model_args=model_args, model_kwargs=model_kwargs,
            use_scan=use_scan)

        attempts.append(result['record'])
        if result['record'].outcome == 'ok':
            diagnostics = result['diagnostics']
            diagnostics.attempts = attempts
            return result['samples'], diagnostics

    raise RuntimeError(
        f'NUTS failed to warm up in {max_attempts} attempts: '
        + '; '.join(f'seed {a.seed}: {a.reason}' for a in attempts))


def _run_one_attempt(potential_fn, init_params, transforms, keep, *,
                     num_warmup, num_samples, seed, attempt, max_tree_depth,
                     target_accept_prob, dense_mass, step_size_floor,
                     check_after, check_window, saturation_fraction,
                     report_every, wall_clock_limit, progress, hmc, jax, jnp,
                     jr, model_args, model_kwargs, use_scan=False,
                     checkpoint=None, checkpoint_every=5000):
    """
    One attempt at a chain; the watchdog lives here.

    Iterations run in a plain Python loop over a jitted ``sample_kernel``.  That
    looks wasteful -- one host round-trip per iteration to read the step size,
    tree depth and acceptance -- and the obvious optimization is to run chunks
    inside a single ``lax.scan``.  Measured on the 15 yr posterior, the scan
    version is *2.7x slower*: each NUTS step already costs tens to hundreds of
    gradient evaluations, so the per-iteration Python overhead is negligible
    beside it, while wrapping a data-dependent tree-doubling loop in a scan
    costs XLA more than it saves.  ``use_scan=True`` keeps that variant
    available for re-benchmarking on other hardware; it is not the default.
    """
    init_kernel, sample_kernel = hmc(potential_fn=potential_fn, algo='NUTS')
    sample_kernel = jax.jit(sample_kernel)

    state = init_kernel(
        init_params, num_warmup=num_warmup, adapt_step_size=True,
        adapt_mass_matrix=True, dense_mass=dense_mass,
        max_tree_depth=max_tree_depth, target_accept_prob=target_accept_prob,
        rng_key=jr.PRNGKey(seed), model_args=model_args,
        model_kwargs=model_kwargs or {})

    @jax.jit
    def constrain(z):
        return {name: transforms[name](z[name]) if name in transforms else z[name]
                for name in keep}

    scan_cache = {}

    def scan_chunk(length, collect_samples):
        key = (length, collect_samples)
        if key not in scan_cache:
            def body(current, _):
                current = sample_kernel(current)
                trace = (current.adapt_state.step_size, current.num_steps,
                         current.mean_accept_prob, current.diverging)
                return current, (trace, constrain(current.z) if collect_samples else {})
            scan_cache[key] = jax.jit(
                lambda start: jax.lax.scan(body, start, None, length=length))
        return scan_cache[key]

    start = time.time()
    warm_step, warm_accept, warm_depth = [], [], []
    collected, sample_depth = {k: [] for k in keep}, []
    divergences = 0
    done = 0
    total = num_warmup + num_samples
    stride = max(1, min(report_every, check_window // 2 or 1)) if use_scan else 1

    while done < total:
        warmup = done < num_warmup
        length = min(stride, (num_warmup if warmup else total) - done)

        if use_scan:
            state, (trace, payload) = scan_chunk(length, not warmup)(state)
            step_sizes, num_steps, accepts, diverging = (
                np.asarray(part) for part in trace)
        else:
            state = sample_kernel(state)
            step_sizes = np.array([float(state.adapt_state.step_size)])
            num_steps = np.array([int(state.num_steps)])
            accepts = np.array([float(state.mean_accept_prob)])
            diverging = np.array([bool(state.diverging)])
            payload = constrain(state.z) if not warmup else {}

        depths = np.ceil(np.log2(np.maximum(num_steps, 1) + 1)).astype(int)
        done += length

        if warmup:
            warm_step.extend(step_sizes.tolist())
            warm_accept.extend(accepts.tolist())
            warm_depth.extend(depths.tolist())
        else:
            divergences += int(np.sum(diverging))
            sample_depth.extend(depths.tolist())
            for name in keep:
                value = np.asarray(payload[name])
                collected[name].append(value if use_scan else value[None])

        step_size, accept, depth = (float(step_sizes[-1]), float(accepts[-1]),
                                    int(depths[-1]))

        info = {
            'phase': 'warmup' if warmup else 'sample',
            'attempt': attempt, 'iteration': done, 'total': total,
            'step_size': step_size, 'accept_prob': accept,
            'tree_depth': depth, 'divergences': divergences,
            'elapsed': time.time() - start,
        }
        if progress is not None and (done % report_every < length or done == total):
            progress(info)

        if (checkpoint is not None and not warmup
                and ((done - num_warmup) % checkpoint_every < length or done == total)):
            checkpoint({k: np.concatenate(v) for k, v in collected.items()}, info)

        if not np.isfinite(step_size) or not np.isfinite(accept):
            return _failed(seed, 'diverged', done, start, step_size, accept,
                           'non-finite step size or acceptance probability')

        # the watchdog: warm-up only, and only once adaptation has had a chance
        if warmup and done >= check_after and done % 10 < length:
            window_step = warm_step[-check_window:]
            window_depth = np.asarray(warm_depth[-check_window:])
            if (float(np.median(window_step)) < step_size_floor
                    and np.mean(window_depth >= max_tree_depth) >= saturation_fraction):
                return _failed(
                    seed, 'collapsed', done, start, step_size, accept,
                    f'step size {np.median(window_step):.2e} below '
                    f'{step_size_floor:g} with tree depth saturated at '
                    f'{max_tree_depth} over the last {check_window} iterations')

        if wall_clock_limit is not None and time.time() - start > wall_clock_limit:
            return _failed(seed, 'timeout', done, start, step_size, accept,
                           f'exceeded {wall_clock_limit:g} s')

    elapsed = time.time() - start
    samples = {k: np.concatenate(v) for k, v in collected.items()}
    diagnostics = SamplerDiagnostics(
        seed=seed, num_warmup=num_warmup, num_samples=num_samples,
        elapsed=elapsed, step_size=float(state.adapt_state.step_size),
        mean_accept_prob=float(state.mean_accept_prob),
        divergences=divergences,
        max_depth_fraction=float(np.mean(np.asarray(sample_depth) >= max_tree_depth))
        if sample_depth else 0.0,
        warmup_step_size=np.asarray(warm_step),
        warmup_accept_prob=np.asarray(warm_accept),
        warmup_tree_depth=np.asarray(warm_depth),
        sample_tree_depth_mean=float(np.mean(sample_depth)) if sample_depth else 0.0)

    return {'record': AttemptRecord(seed, 'ok', total, elapsed,
                                    diagnostics.step_size,
                                    diagnostics.mean_accept_prob),
            'samples': samples, 'diagnostics': diagnostics}


def _failed(seed, outcome, iterations, start, step_size, accept, reason):
    return {'record': AttemptRecord(seed, outcome, iterations,
                                    time.time() - start, step_size, accept,
                                    reason),
            'samples': None, 'diagnostics': None}


def effective_sample_size(chain):
    """
    Effective sample size of a 1-D chain, by the initial-positive-sequence rule.

    Parameters
    ----------
    chain : array_like, shape (N,)

    Returns
    -------
    float
    """
    x = np.asarray(chain, dtype=np.float64)
    n = x.size
    if n < 4:
        return float(n)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var == 0:
        return float(n)

    padded = np.zeros(2 * n)
    padded[:n] = x
    spectrum = np.fft.rfft(padded)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum))[:n] / n
    rho = acov / acov[0]

    total, k = 0.0, 1
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        total += pair
        k += 2
    return float(n / (1.0 + 2.0 * total))
