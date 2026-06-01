
"""
STEP 6 — MARKETING SPEND OPTIMIZATION: LKR 5M for Western Province
=================================================================

Purpose:
    Allocate LKR 5,000,000 across Western Province outlets to
    maximize expected incremental sales volume.

Approach:
    1. Greedy marginal ROI optimization
    2. Redistribute any remaining budget proportionally
       to incremental potential
    3. Guarantee total allocation = exactly LKR 5,000,000

Run:
    python 06_budget_optimization.py
"""

import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BUDGET] %(message)s"
)

log = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

GOLD_DIR = "gold"
OUTPUTS_DIR = "outputs"

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# =============================================================================
# PARAMETERS
# =============================================================================

TOTAL_BUDGET = 5_000_000
PROVINCE_TARGET = "Western"

MIN_SPEND = 5_000
MAX_SPEND = 200_000

EFFICIENCY_BETA = 0.6

SPEND_CATEGORIES = {
    "discount": 0.40,
    "merchandising": 0.35,
    "cooler_support": 0.25
}

# =============================================================================
# RESPONSE FUNCTIONS
# =============================================================================

def volume_gain(spend, alpha, beta=EFFICIENCY_BETA):
    """
    Cobb-Douglas diminishing returns function
    """
    return alpha * (spend ** beta)


def marginal_gain_per_lkr(spend, alpha, beta=EFFICIENCY_BETA):
    """
    d(gain)/d(spend)
    """

    if spend <= 0:
        spend = MIN_SPEND

    return alpha * beta * (spend ** (beta - 1))


# =============================================================================
# ALPHA CALCULATION
# =============================================================================

def compute_alpha(df):
    """
    Compute outlet response coefficient alpha.
    """

    inc = df["incremental_potential"].clip(lower=0)

    total_inc = inc.sum()

    if total_inc <= 0:
        return pd.Series(
            np.ones(len(df)) * 1e-6,
            index=df.index
        )

    alpha = (
        (inc / total_inc)
        * (total_inc / (TOTAL_BUDGET ** EFFICIENCY_BETA))
    )

    return alpha.clip(lower=1e-8)


# =============================================================================
# GREEDY OPTIMIZATION
# =============================================================================

def greedy_allocation(df):

    log.info("Running greedy allocation...")

    df = df.copy()

    n = len(df)

    alloc = np.zeros(n)

    alpha = compute_alpha(df).values

    budget_remaining = TOTAL_BUDGET

    eligible = (
        df["incremental_potential"]
        .fillna(0)
        .values > 0
    )

    n_eligible = eligible.sum()

    min_required = n_eligible * MIN_SPEND

    # -------------------------------------------------------------------------
    # Give minimum spend first
    # -------------------------------------------------------------------------

    if min_required > budget_remaining:

        top_n = int(budget_remaining // MIN_SPEND)

        top_idx = np.argsort(
            -df["incremental_potential"].values
        )[:top_n]

        alloc[top_idx] = MIN_SPEND

        budget_remaining = 0

    else:

        alloc[eligible] = MIN_SPEND

        budget_remaining -= min_required

    # -------------------------------------------------------------------------
    # Greedy ROI allocation
    # -------------------------------------------------------------------------

    step = MIN_SPEND

    while budget_remaining >= step:

        mgains = np.array([
            marginal_gain_per_lkr(
                alloc[i],
                alpha[i]
            ) if eligible[i] else -1
            for i in range(n)
        ])

        mgains[alloc >= MAX_SPEND] = -1

        best_idx = np.argmax(mgains)

        if mgains[best_idx] <= 0:
            break

        alloc[best_idx] += step

        budget_remaining -= step

    df["Trade_Spend_LKR"] = alloc

    return df


# =============================================================================
# REDISTRIBUTE REMAINING BUDGET
# =============================================================================

def redistribute_remaining_budget(
    df,
    total_budget=TOTAL_BUDGET
):
    """
    Ensure the entire budget is allocated.
    """

    df = df.copy()

    spent = df["Trade_Spend_LKR"].sum()

    remaining = total_budget - spent

    if remaining <= 0:
        return df

    log.info(
        f"Redistributing remaining budget: "
        f"LKR {remaining:,.2f}"
    )

    weights = (
        df["incremental_potential"]
        .clip(lower=0)
        .fillna(0)
    )

    total_weight = weights.sum()

    if total_weight <= 0:
        return df

    weights = weights / total_weight

    extra_alloc = weights * remaining

    df["Trade_Spend_LKR"] += extra_alloc

    # Final adjustment to make exactly 5M

    diff = total_budget - df["Trade_Spend_LKR"].sum()

    if abs(diff) > 0.01:

        idx = df["incremental_potential"].idxmax()

        df.loc[idx, "Trade_Spend_LKR"] += diff

    return df


# =============================================================================
# RECALCULATE KPI
# =============================================================================

def calculate_metrics(df):

    df = df.copy()

    alpha = compute_alpha(df).values

    df["expected_volume_gain_L"] = [
        volume_gain(
            df.iloc[i]["Trade_Spend_LKR"],
            alpha[i]
        )
        for i in range(len(df))
    ]

    df["roi_litres_per_1000_lkr"] = np.where(
        df["Trade_Spend_LKR"] > 0,
        df["expected_volume_gain_L"]
        / (df["Trade_Spend_LKR"] / 1000),
        0
    )

    return df


# =============================================================================
# DISTRIBUTOR SUMMARY
# =============================================================================

def distributor_summary(df):

    summary = (
        df.groupby("Distributor_ID")
        .agg(
            total_spend_lkr=(
                "Trade_Spend_LKR",
                "sum"
            ),
            n_outlets_supported=(
                "Trade_Spend_LKR",
                lambda x: (x > 0).sum()
            ),
            total_volume_gain_L=(
                "expected_volume_gain_L",
                "sum"
            ),
            avg_roi=(
                "roi_litres_per_1000_lkr",
                "mean"
            ),
            total_potential_L=(
                "incremental_potential",
                "sum"
            )
        )
        .reset_index()
    )

    summary["budget_share_pct"] = (
        summary["total_spend_lkr"]
        / TOTAL_BUDGET
        * 100
    ).round(2)

    return summary


# =============================================================================
# MAIN
# =============================================================================

def run():

    input_file = os.path.join(
        GOLD_DIR,
        "predictions_full.csv"
    )

    preds = pd.read_csv(input_file)

    western = preds[
        preds["Province"] == PROVINCE_TARGET
    ].copy()

    log.info(
        f"Western Province outlets: {len(western):,}"
    )

    if len(western) == 0:

        log.error(
            "No Western Province outlets found."
        )

        return

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    western_alloc = greedy_allocation(western)

    western_alloc = redistribute_remaining_budget(
        western_alloc,
        TOTAL_BUDGET
    )

    western_alloc = calculate_metrics(
        western_alloc
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    total_spent = (
        western_alloc["Trade_Spend_LKR"]
        .sum()
    )

    total_gain = (
        western_alloc["expected_volume_gain_L"]
        .sum()
    )

    supported = (
        western_alloc["Trade_Spend_LKR"] > 0
    ).sum()

    log.info(
        f"Total spent: LKR {total_spent:,.2f}"
    )

    log.info(
        f"Outlets supported: {supported}"
    )

    log.info(
        f"Expected volume gain: "
        f"{total_gain:,.2f} L"
    )

    log.info(
        f"ROI: "
        f"{total_gain/(total_spent/1000):.2f}"
        f" L per 1,000 LKR"
    )

    # -------------------------------------------------------------------------
    # Spend categories
    # -------------------------------------------------------------------------

    for cat, share in SPEND_CATEGORIES.items():

        western_alloc[
            f"spend_{cat}_lkr"
        ] = (
            western_alloc["Trade_Spend_LKR"]
            * share
        ).round(2)

    # -------------------------------------------------------------------------
    # Distributor summary
    # -------------------------------------------------------------------------

    dist_summary = distributor_summary(
        western_alloc
    )

    dist_summary.to_csv(
        os.path.join(
            OUTPUTS_DIR,
            "distributor_budget_summary.csv"
        ),
        index=False
    )

    # -------------------------------------------------------------------------
    # Submission file
    # -------------------------------------------------------------------------

    submission = western_alloc[
        [
            "Outlet_ID",
            "Trade_Spend_LKR"
        ]
    ].rename(
        columns={
            "Trade_Spend_LKR":
            "Trade Spend Allocation (LKR)"
        }
    )

    submission.to_csv(
        os.path.join(
            OUTPUTS_DIR,
            "cyberRiders_budget_allocations.csv"
        ),
        index=False
    )

    # -------------------------------------------------------------------------
    # Full allocation file
    # -------------------------------------------------------------------------

    western_alloc.to_csv(
        os.path.join(
            GOLD_DIR,
            "western_allocations_full.csv"
        ),
        index=False
    )

    log.info(
        "Saved: outputs/cyberRiders_budget_allocations.csv"
    )

    log.info(
        f"Final Budget Allocated: "
        f"LKR {submission['Trade_Spend_Allocation_LKR'].sum():,.2f}"
    )

    return western_alloc, dist_summary


if __name__ == "__main__":
    run()