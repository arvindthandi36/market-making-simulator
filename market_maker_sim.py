"""
Market-Making Simulator
========================

A single-file simulation of a market maker quoting two-sided prices around a
mid that follows geometric Brownian motion (GBM). The MM:

  * Posts a bid and ask around the mid with a configurable base spread.
  * Skews its quotes against its inventory so it leans away from building a
    large one-sided position (inventory risk management).
  * Gets filled by two kinds of flow each step:
      - NOISE (uninformed) flow, arriving independently of the next price move,
        which lets the MM earn the spread. Its arrival rate decays with how far
        the quote sits from the mid, so tighter quotes fill more often and the
        inventory skew actually steers flow and mean-reverts the position.
      - INFORMED flow, arriving only when the price trades THROUGH the quote,
        which is the adverse-selection cost the MM pays.

Outputs:
  * A 3-panel diagnostic figure:
      1. Price path with MM bid/ask, fill markers, and a zoomed inset that
         resolves the (tiny) bid-ask band.
      2. Inventory, autoscaled to its actual range.
      3. Cumulative PnL decomposed into spread-capture and inventory
         components alongside the mark-to-market total.
    A parameter box makes the figure self-documenting.
  * A Monte Carlo figure: the distribution of session PnL over many independent
    paths, with win rate and the session-level Sharpe.
  * Summary stats printed to stdout.

Note on Sharpe: a session-level Sharpe (mean session PnL / std of session PnL
across Monte Carlo paths) is the meaningful number. Annualizing minute-level
PnL over a single 5-day path is a meaningless extrapolation and is labeled as
such where shown.

Requires: numpy, matplotlib.
"""

from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# --- Price process (GBM): dS = mu*S*dt + sigma*S*dW ---
S0       = 100.0     # initial underlying price
MU       = 0.05      # annual drift
SIGMA    = 0.20      # annual volatility
DT       = 1.0 / 252 / 390   # time step (1 minute, given 390 min/trading day)
HORIZON  = 5.0 / 252         # total simulated time (5 trading days)

# --- Market maker quoting ---
BASE_SPREAD   = 0.10     # full base spread in price units (half on each side)
SKEW_PER_UNIT = 0.004    # how much the mid quote is shifted per unit inventory
QUOTE_SIZE    = 1.0      # size posted on each side (units filled per hit)

# --- Fill model ---
# Each side can be hit by two independent flows per step:
#
#   * NOISE (uninformed) flow. Arrives independently of the next price move, so
#     it lets the MM earn the spread without adverse selection. Its arrival
#     probability is NOISE_LIQUIDITY at a quote sitting exactly on the mid and
#     decays exponentially (rate NOISE_DECAY) with the quote's distance from the
#     mid -- so a tighter quote fills more often. This distance dependence is
#     what makes the inventory skew actually pull flow to the heavy side and
#     mean-revert the position.
#
#   * INFORMED flow. Arrives only when the price trades THROUGH the quote; its
#     probability rises with penetration depth (rate FILL_SENSITIVITY). These
#     fills are systematically on the wrong side of the next move -- the
#     adverse-selection cost. With noise flow sized above this cost, the MM is
#     net profitable while inventory stays bounded by the skew.
NOISE_LIQUIDITY  = 0.18   # baseline noise fill prob per side at zero distance
NOISE_DECAY      = 12.0   # decay of noise arrival per unit of quote distance
FILL_SENSITIVITY = 8.0    # informed-fill sensitivity to penetration depth

# --- Inventory limits ---
MAX_INVENTORY = 50.0   # MM stops quoting the side that would breach this

# --- Monte Carlo ---
N_PATHS = 1000   # independent random paths for the session-PnL distribution

# --- Sensitivity sweep ---
# Paths per Monte Carlo at each swept parameter value. Defaults to the full
# N_PATHS so every point is a complete Monte Carlo; lower it to trade
# resolution for speed (the whole sweep is ~SWEEP_PATHS * 28 sessions).
SWEEP_PATHS = N_PATHS

# --- Reproducibility ---
SEED = 42

# --- Plotting ---
INSET_STEPS = 60   # width (in steps) of the zoomed price-panel inset


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(seed=SEED, collect_paths=True):
    """Run one market-making session.

    seed          : RNG seed, so independent paths can be drawn for Monte Carlo.
    collect_paths : when False, skips building the per-fill marker logs (used by
                    the Monte Carlo, which only needs aggregate PnL).
    """
    rng = np.random.default_rng(seed)

    n_steps = int(HORIZON / DT)
    times = np.arange(n_steps + 1) * DT

    # --- Generate the GBM mid-price path ---
    # S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    shocks = rng.standard_normal(n_steps)
    drift = (MU - 0.5 * SIGMA**2) * DT
    diffusion = SIGMA * np.sqrt(DT) * shocks
    log_returns = drift + diffusion
    mid = np.empty(n_steps + 1)
    mid[0] = S0
    mid[1:] = S0 * np.exp(np.cumsum(log_returns))

    # --- Storage ---
    bids = np.full(n_steps + 1, np.nan)
    asks = np.full(n_steps + 1, np.nan)
    inventory = np.zeros(n_steps + 1)
    cash = np.zeros(n_steps + 1)
    pnl = np.zeros(n_steps + 1)

    # PnL decomposition (instrumentation only — does not affect the model):
    #   spread_pnl[k]    cumulative edge captured at fills vs. the quote-time mid
    #   inventory_pnl[k] cumulative inventory * change-in-mid
    # These two are constructed to sum exactly to the mark-to-market `pnl`.
    spread_pnl = np.zeros(n_steps + 1)
    inventory_pnl = np.zeros(n_steps + 1)

    # Fill log for plotting trade markers: (time index, executed price).
    buy_fills = []   # bid hit  -> we bought
    sell_fills = []  # ask hit  -> we sold

    half_spread = BASE_SPREAD / 2.0

    inv = 0.0
    cash_bal = 0.0
    sc = 0.0   # running spread-capture PnL
    ip = 0.0   # running inventory PnL

    for t in range(n_steps):
        m = mid[t]

        # Inventory skew: when long (inv > 0) we shift our quotes DOWN so we
        # are more likely to sell (ask closer to mid) and less likely to buy
        # (bid further from mid), nudging inventory back toward zero.
        skew = SKEW_PER_UNIT * inv
        quote_mid = m - skew

        bid = quote_mid - half_spread
        ask = quote_mid + half_spread

        # Respect inventory limits: don't post a side that would push us past
        # the cap.
        post_bid = inv < MAX_INVENTORY
        post_ask = inv > -MAX_INVENTORY

        bids[t] = bid if post_bid else np.nan
        asks[t] = ask if post_ask else np.nan

        # Price move over this step drives the informed component of fills.
        next_mid = mid[t + 1]
        move = next_mid - m

        # Distances of each quote from the mid. The skew makes one side tighter
        # (smaller distance -> more noise flow) and the other wider.
        ask_dist = ask - m          # = half_spread - skew
        bid_dist = m - bid          # = half_spread + skew

        # --- Ask fill: noise buyers lift it, plus informed flow if price
        #     trades through it (next_mid above the ask). ---
        if post_ask:
            # Noise flow: independent of the move, decays with quote distance.
            p_noise = NOISE_LIQUIDITY * np.exp(-NOISE_DECAY * max(ask_dist, 0.0))
            # Informed flow: only when the price penetrates above the ask.
            depth = max(next_mid - ask, 0.0)
            p_informed = 1.0 - np.exp(-FILL_SENSITIVITY * depth)
            # Probability of at least one of the two independent flows hitting.
            prob = p_noise + p_informed - p_noise * p_informed
            if rng.random() < prob:
                inv -= QUOTE_SIZE          # we sold
                cash_bal += ask * QUOTE_SIZE
                sc += (ask - m) * QUOTE_SIZE        # edge earned vs. mid
                if collect_paths:
                    sell_fills.append((t, ask))

        # --- Bid fill: noise sellers hit it, plus informed flow if price
        #     trades through it (next_mid below the bid). ---
        if post_bid:
            p_noise = NOISE_LIQUIDITY * np.exp(-NOISE_DECAY * max(bid_dist, 0.0))
            depth = max(bid - next_mid, 0.0)
            p_informed = 1.0 - np.exp(-FILL_SENSITIVITY * depth)
            prob = p_noise + p_informed - p_noise * p_informed
            if rng.random() < prob:
                inv += QUOTE_SIZE          # we bought
                cash_bal -= bid * QUOTE_SIZE
                sc += (m - bid) * QUOTE_SIZE        # edge earned vs. mid
                if collect_paths:
                    buy_fills.append((t, bid))

        inventory[t + 1] = inv
        cash[t + 1] = cash_bal
        # Mark-to-market PnL: realized cash + inventory valued at current mid.
        pnl[t + 1] = cash_bal + inv * next_mid

        # PnL decomposition. Inventory PnL uses the post-fill inventory carried
        # into this step's price move, matching how `pnl` marks to next_mid.
        ip += inv * move
        spread_pnl[t + 1] = sc
        inventory_pnl[t + 1] = ip

    # Carry the final quotes for plotting continuity.
    bids[-1] = bids[-2]
    asks[-1] = asks[-2]

    # The two components must reconstruct the mark-to-market PnL exactly.
    assert np.allclose(spread_pnl + inventory_pnl, pnl), \
        "PnL decomposition does not sum to the mark-to-market total"

    buy_fills = np.array(buy_fills).reshape(-1, 2)
    sell_fills = np.array(sell_fills).reshape(-1, 2)

    return {
        "times": times,
        "mid": mid,
        "bids": bids,
        "asks": asks,
        "inventory": inventory,
        "pnl": pnl,
        "spread_pnl": spread_pnl,
        "inventory_pnl": inventory_pnl,
        "buy_fills": buy_fills,
        "sell_fills": sell_fills,
    }


# ---------------------------------------------------------------------------
# Stats & plotting
# ---------------------------------------------------------------------------

def summary_stats(res):
    pnl = res["pnl"]
    inv = res["inventory"]

    final_pnl = pnl[-1]
    max_abs_inv = np.max(np.abs(inv))

    # Per-path "annualized" Sharpe of step PnL. KEPT ONLY FOR REFERENCE and
    # explicitly NOT MEANINGFUL: annualizing minute-level PnL by sqrt(steps/yr)
    # over a single 5-day window extrapolates ~50,000x and is dominated by the
    # one path's inventory swings. Use the session-level Sharpe from the Monte
    # Carlo instead (see monte_carlo()).
    pnl_changes = np.diff(pnl)
    steps_per_year = 1.0 / DT
    mean = pnl_changes.mean()
    std = pnl_changes.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(steps_per_year) if std > 0 else float("nan")

    return {
        "final_pnl": final_pnl,
        "max_abs_inventory": max_abs_inv,
        "sharpe_per_path_annualized": sharpe,   # not meaningful; see note above
    }


def monte_carlo(n_paths=N_PATHS, base_seed=SEED):
    """Run many independent sessions and summarize the session-PnL distribution.

    Each path uses a distinct seed (base_seed + 1 + i) so the runs are
    independent of one another and of the single displayed path (which uses
    base_seed). The session-level Sharpe is the meaningful risk-adjusted number:
    mean session PnL divided by the std of session PnL across paths.
    """
    final_pnls = np.empty(n_paths)
    max_invs = np.empty(n_paths)
    for i in range(n_paths):
        res = simulate(seed=base_seed + 1 + i, collect_paths=False)
        final_pnls[i] = res["pnl"][-1]
        max_invs[i] = np.max(np.abs(res["inventory"]))

    mean = final_pnls.mean()
    std = final_pnls.std(ddof=1)
    session_sharpe = mean / std if std > 0 else float("nan")
    win_rate = 100.0 * np.mean(final_pnls > 0)

    return {
        "final_pnls": final_pnls,
        "max_invs": max_invs,
        "mean_pnl": mean,
        "std_pnl": std,
        "session_sharpe": session_sharpe,
        "win_rate": win_rate,
        "n_paths": n_paths,
    }


def _scatter_fills(ax, days, fills, color, marker, label):
    """Plot trade markers from a fill log of (time index, price) rows."""
    if fills.size == 0:
        return
    idx = fills[:, 0].astype(int)
    ax.scatter(
        days[idx], fills[:, 1], s=18, c=color, marker=marker,
        edgecolors="black", linewidths=0.3, zorder=5, label=label,
    )


def plot(res, stats):
    times = res["times"]
    # Convert time axis to "trading days" for readability.
    days = times * 252

    mid = res["mid"]
    bids = res["bids"]
    asks = res["asks"]
    buy_fills = res["buy_fills"]
    sell_fills = res["sell_fills"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    # --- Panel 1: price path with fills + zoomed inset for the quotes ---
    ax = axes[0]
    ax.plot(days, mid, color="black", lw=1.0, label="Mid", zorder=2)
    _scatter_fills(ax, days, buy_fills, "tab:green", "^", "Bought (bid hit)")
    _scatter_fills(ax, days, sell_fills, "tab:red", "v", "Sold (ask hit)")
    ax.set_ylabel("Price")
    ax.set_title("Underlying mid-price (GBM) with MM fills")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    ax.grid(alpha=0.3)

    # Parameter box — makes the figure self-documenting.
    params = (
        f"S0      = {S0:.0f}\n"
        f"mu      = {MU:.2f}\n"
        f"sigma   = {SIGMA:.2f}\n"
        f"spread  = {BASE_SPREAD:.2f}\n"
        f"skew/u  = {SKEW_PER_UNIT:.3f}\n"
        f"horizon = {HORIZON * 252:.0f}d"
    )
    ax.text(
        0.995, 0.97, params, transform=ax.transAxes, ha="right", va="top",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )

    # Zoomed inset over the first INSET_STEPS steps, where the bid/mid/ask
    # separation (~half the spread each side) is actually resolvable.
    w = min(INSET_STEPS, len(days) - 1)
    sl = slice(0, w + 1)
    axins = ax.inset_axes([0.06, 0.10, 0.40, 0.45])
    axins.fill_between(
        days[sl], bids[sl], asks[sl], color="tab:blue", alpha=0.20,
        label="Bid-ask band",
    )
    axins.plot(days[sl], mid[sl], color="black", lw=1.0, label="Mid")
    axins.plot(days[sl], asks[sl], color="tab:red", lw=0.9, label="Ask")
    axins.plot(days[sl], bids[sl], color="tab:green", lw=0.9, label="Bid")
    # Fills that fall inside the inset window.
    if buy_fills.size:
        bf = buy_fills[buy_fills[:, 0] < w]
        _scatter_fills(axins, days, bf, "tab:green", "^", None)
    if sell_fills.size:
        sf = sell_fills[sell_fills[:, 0] < w]
        _scatter_fills(axins, days, sf, "tab:red", "v", None)
    axins.set_title(f"First {w} steps (zoom)", fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(alpha=0.3)
    axins.legend(loc="upper left", fontsize=6, ncol=2)
    ax.indicate_inset_zoom(axins, edgecolor="gray", alpha=0.6)

    # --- Panel 2: inventory (autoscaled to its actual range) ---
    ax = axes[1]
    inv = res["inventory"]
    ax.plot(days, inv, color="tab:blue", lw=1.0)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    # Only show the inventory caps if we get close enough for them to matter.
    if np.max(np.abs(inv)) >= 0.8 * MAX_INVENTORY:
        ax.axhline(MAX_INVENTORY, color="red", lw=0.6, ls=":", alpha=0.6,
                   label="Inventory cap")
        ax.axhline(-MAX_INVENTORY, color="red", lw=0.6, ls=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8)
    # Autoscale with a small symmetric pad around the realized range.
    lo, hi = inv.min(), inv.max()
    pad = max(1.0, 0.15 * (hi - lo))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Inventory (units)")
    ax.set_title("Market maker inventory")
    ax.grid(alpha=0.3)

    # --- Panel 3: PnL decomposition ---
    ax = axes[2]
    ax.plot(days, res["pnl"], color="tab:purple", lw=1.4, label="Total (MtM)")
    ax.plot(days, res["spread_pnl"], color="tab:green", lw=1.0,
            label="Spread capture")
    ax.plot(days, res["inventory_pnl"], color="tab:orange", lw=1.0,
            label="Inventory PnL")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("PnL")
    ax.set_xlabel("Trading days")
    ax.set_title("Cumulative PnL decomposition")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    txt = (
        f"Single-path PnL : {stats['final_pnl']:.2f}\n"
        f"Max |inv|       : {stats['max_abs_inventory']:.0f}\n"
        f"Session Sharpe  : {stats['session_sharpe']:.2f}  (MC, n={N_PATHS})\n"
        f"Win rate        : {stats['win_rate']:.0f}%  (MC)"
    )
    ax.text(
        0.99, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()
    return fig


def plot_mc(mc):
    """Histogram of session PnL across the Monte Carlo paths."""
    pnls = mc["final_pnls"]
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(pnls, bins=40, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="gray", lw=1.0, ls="--", label="break-even")
    ax.axvline(mc["mean_pnl"], color="tab:red", lw=1.5,
               label=f"mean = {mc['mean_pnl']:.2f}")
    ax.set_xlabel("Session PnL")
    ax.set_ylabel("Number of paths")
    ax.set_title(f"Monte Carlo session-PnL distribution ({mc['n_paths']} paths)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    txt = (
        f"Mean PnL       : {mc['mean_pnl']:.2f}\n"
        f"Std PnL        : {mc['std_pnl']:.2f}\n"
        f"Win rate       : {mc['win_rate']:.1f}%\n"
        f"Session Sharpe : {mc['session_sharpe']:.2f}"
    )
    ax.text(
        0.015, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

@contextmanager
def _override_globals(**overrides):
    """Temporarily override module-level parameters, then restore them.

    This lets the sweep re-run the existing simulate()/monte_carlo() at
    different parameter values without touching their internals or permanently
    changing the baseline globals.
    """
    g = globals()
    saved = {k: g[k] for k in overrides}
    g.update(overrides)
    try:
        yield
    finally:
        g.update(saved)


def _zero_crossings(values, means):
    """Linearly-interpolated parameter values where mean PnL crosses zero."""
    values = np.asarray(values, float)
    means = np.asarray(means, float)
    crossings = []
    for i in range(len(values) - 1):
        a, b = means[i], means[i + 1]
        if a == 0.0:
            crossings.append(float(values[i]))
        elif a * b < 0.0:                       # sign change between the points
            xc = values[i] + (0.0 - a) * (values[i + 1] - values[i]) / (b - a)
            crossings.append(float(xc))
    return crossings


# Each entry: the global to vary, a human-readable label, and the values to
# sweep. All four bracket the baseline and extend into plausibly loss-making
# territory so any profit/loss crossover is captured.
SWEEP_SPECS = [
    ("NOISE_LIQUIDITY", "Noise liquidity",
     [0.02, 0.05, 0.09, 0.13, 0.18, 0.24, 0.30]),
    ("FILL_SENSITIVITY", "Informed-fill sensitivity",
     [0.0, 3.0, 6.0, 8.0, 12.0, 20.0, 32.0]),
    ("BASE_SPREAD", "Quoted spread",
     [0.02, 0.05, 0.08, 0.10, 0.14, 0.20, 0.28]),
    ("MAX_INVENTORY", "Inventory cap",
     [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 50.0]),
]


def sensitivity_analysis(n_paths=SWEEP_PATHS, base_seed=SEED):
    """Sweep each key parameter one at a time, running a full Monte Carlo at
    every value while holding all others at baseline.

    Each Monte Carlo reuses the same base_seed, so every value sees the same
    set of random paths (common random numbers) and differences in mean PnL
    reflect the parameter, not sampling noise.
    """
    results = []
    for name, label, values in SWEEP_SPECS:
        baseline = globals()[name]
        means, wins = [], []
        for v in values:
            with _override_globals(**{name: v}):
                mc = monte_carlo(n_paths, base_seed)
            means.append(mc["mean_pnl"])
            wins.append(mc["win_rate"])
        means = np.array(means)
        results.append({
            "name": name,
            "label": label,
            "values": np.array(values, float),
            "means": means,
            "wins": np.array(wins),
            "baseline": baseline,
            "crossings": _zero_crossings(values, means),
            "pnl_range": float(means.max() - means.min()),
        })
    return results


def print_sensitivity_summary(results):
    """Rank parameters by how strongly they move mean session PnL."""
    ranked = sorted(results, key=lambda r: r["pnl_range"], reverse=True)

    print("=" * 64)
    print("Sensitivity summary (one-at-a-time sweeps, ranked by PnL swing)")
    print("=" * 64)
    for r in ranked:
        if r["crossings"]:
            # Crossing nearest the baseline = how far you can move before losses.
            nearest = min(r["crossings"], key=lambda x: abs(x - r["baseline"]))
            margin = (100.0 * abs(nearest - r["baseline"]) / abs(r["baseline"])
                      if r["baseline"] else float("nan"))
            cross_txt = (f"break-even @ {nearest:.3g} "
                         f"(baseline {r['baseline']:.3g}, {margin:.0f}% away)")
        else:
            cross_txt = "no break-even in range (robust)"
        print(f"  {r['label']:<26s} PnL swing {r['pnl_range']:6.2f}  | {cross_txt}")
    print("-" * 64)
    print(f"  Most sensitive : {ranked[0]['label']}")
    print(f"  Least sensitive: {ranked[-1]['label']}")
    print("=" * 64)


def plot_sensitivity(results):
    """2x2 grid: mean session PnL (left axis) and win rate (right axis) vs each
    swept parameter, with the loss-making region shaded and break-even marked.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax1, r in zip(axes.flat, results):
        values, means, wins = r["values"], r["means"], r["wins"]

        # Shade where the (interpolated) mean PnL is negative = loss-making.
        dense = np.linspace(values.min(), values.max(), 400)
        dmean = np.interp(dense, values, means)
        ax1.fill_between(dense, 0, 1, where=dmean < 0,
                         transform=ax1.get_xaxis_transform(),
                         color="tab:red", alpha=0.08, zorder=0,
                         label="loss-making")

        # Left axis: mean session PnL.
        ax1.plot(values, means, "o-", color="tab:blue", label="Mean PnL")
        ax1.axhline(0, color="gray", lw=0.8, ls="--")
        ax1.set_xlabel(r["label"])
        ax1.set_ylabel("Mean session PnL", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title(r["label"])
        ax1.grid(alpha=0.3)

        # Right axis: win rate.
        ax2 = ax1.twinx()
        ax2.plot(values, wins, "s--", color="tab:orange", alpha=0.85,
                 label="Win rate")
        ax2.axhline(50, color="tab:orange", lw=0.6, ls=":", alpha=0.6)
        ax2.set_ylabel("Win rate (%)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax2.set_ylim(0, 100)

        # Baseline value and break-even crossing(s).
        ax1.axvline(r["baseline"], color="black", lw=0.8, ls="-.",
                    alpha=0.6, label="baseline")
        for xc in r["crossings"]:
            ax1.axvline(xc, color="tab:red", lw=1.4)
            ax1.annotate(f"break-even\n{xc:.3g}", xy=(xc, 0),
                         xytext=(4, 8), textcoords="offset points",
                         fontsize=7, color="tab:red")

        # Combined legend (lines from both axes plus the shaded region).
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best")

    fig.suptitle("One-at-a-time parameter sensitivity "
                 f"(full Monte Carlo, {SWEEP_PATHS} paths/point)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # One displayed path...
    res = simulate(seed=SEED)
    stats = summary_stats(res)

    # ...and a Monte Carlo over many independent paths for the distribution.
    mc = monte_carlo(N_PATHS, SEED)
    stats["session_sharpe"] = mc["session_sharpe"]
    stats["win_rate"] = mc["win_rate"]

    print("=" * 52)
    print("Market-Making Simulation — Summary")
    print("=" * 52)
    print(f"Steps per session     : {len(res['times']) - 1}")
    print(f"Single-path PnL       : {stats['final_pnl']:.2f}")
    print(f"Single-path max |inv| : {stats['max_abs_inventory']:.0f}")
    print("-" * 52)
    print(f"Monte Carlo paths     : {mc['n_paths']}")
    print(f"Mean session PnL      : {mc['mean_pnl']:.2f}")
    print(f"Std session PnL       : {mc['std_pnl']:.2f}")
    print(f"Win rate              : {mc['win_rate']:.1f}%")
    print(f"Session Sharpe        : {mc['session_sharpe']:.2f}")
    print("-" * 52)
    print(f"Per-path ann. Sharpe  : {stats['sharpe_per_path_annualized']:.2f}"
          "  (NOT meaningful over 5d)")
    print("=" * 52)

    fig = plot(res, stats)
    fig.savefig("market_maker_sim.png", dpi=120)
    print("Saved diagnostic plot to market_maker_sim.png")

    figmc = plot_mc(mc)
    figmc.savefig("market_maker_mc.png", dpi=120)
    print("Saved Monte Carlo plot to market_maker_mc.png")

    # Sensitivity sweeps: full Monte Carlo at each value of each key parameter.
    print(f"\nRunning sensitivity sweeps "
          f"({SWEEP_PATHS} paths/point, this takes a bit)...")
    sens = sensitivity_analysis()
    print_sensitivity_summary(sens)

    figsens = plot_sensitivity(sens)
    figsens.savefig("market_maker_sensitivity.png", dpi=120)
    print("Saved sensitivity plot to market_maker_sensitivity.png")

    plt.show()


if __name__ == "__main__":
    main()
