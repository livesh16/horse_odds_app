#!/usr/bin/env python3
"""
adjust_odds.py

Adjust horse racing odds using features and overround.
Input & output use integer profit-style odds (like Port Louis: 220 = bet 100, win 220).
"""

import argparse
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Use DEBUG to see everything
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

EPS = 1e-12

# Default configuration
DEFAULT_BETAS = {
    "jockey": 0.25,
    "trainer": 0.20,
    "draw": 0.06,
    "going": 0.10,
    "pace": 0.12
}
DEFAULT_W_MARKET = 0.6
DEFAULT_ALPHA = 0.97
DEFAULT_SHRINK = 1.0
DEFAULT_ROUND_STEP = 10
DEFAULT_DECIMALS = 3

# ---------------------------
# Utility functions
# ---------------------------
def safe_log(x):
    return np.log(np.clip(x, EPS, 1.0 - EPS))

def normalize(arr):
    s = np.sum(arr)
    if s <= 0:
        return np.full_like(arr, 1.0 / len(arr))
    return arr / s

def round_to_step(arr, step):
    return np.round(arr / step) * step

# ---------------------------
# Core functions
# ---------------------------
def portlouis_to_market_prob(df):
    # Convert PortLouisOdds to implied market probability
    odds = pd.to_numeric(df["PortLouisOdds"], errors="coerce").fillna(0.0)
    return (100.0 / odds).astype(float)

def compute_race_overrounds(df, p_market):
    # Use Overround if present, otherwise compute sum of p_market per race
    if "Overround" in df.columns:
        return pd.to_numeric(df["Overround"], errors="coerce").fillna(1.0)
    else:
        tmp = df.copy()
        tmp["_p_market_tmp"] = p_market
        return tmp.groupby("RaceID")["_p_market_tmp"].transform("sum")

def apply_feature_adjustments(p_vals, group_df, betas):
    logp = safe_log(p_vals)
    feature_map = {
        "jockey": "JockeyScore",
        "trainer": "TrainerScore",
        "draw": "DrawScore",
        "going": "GoingScore",
        "pace": "PaceScore"
    }
    for key, col in feature_map.items():
        if key in betas and col in group_df.columns:
            x = pd.to_numeric(group_df[col], errors="coerce").fillna(0.5).values.astype(float)
            centered = x - 0.5
            logp += betas[key] * centered
    return np.exp(logp)  # un-normalized

# ---------------------------
# Main function
# ---------------------------
def main(input_csv, output_csv, w_market, alpha, shrink, round_step, decimals, betas=None):
    if betas is None:
        betas = DEFAULT_BETAS.copy()
    betas = {k: v * shrink for k, v in betas.items()}

    df = pd.read_csv(input_csv)
    logger.debug(f"Read CSV:\n{df.head()}")

    logger.debug(f"Parameters in script: w_market={w_market}, alpha={alpha}, shrink={shrink}")

    if "RaceID" not in df.columns or "Horse" not in df.columns or "PortLouisOdds" not in df.columns:
        raise ValueError("Input CSV must contain 'RaceID', 'Horse', and 'PortLouisOdds' columns.")

    # Step 1: Market probabilities from PortLouisOdds
    df["p_market_raw"] = portlouis_to_market_prob(df)

    # Step 2: De-vigged market probabilities
    df["p_true"] = df.groupby("RaceID")["p_market_raw"].transform(lambda s: s / s.sum())

    # Step 3: Per-race overround
    df["race_overround"] = compute_race_overrounds(df, df["p_market_raw"].values)

    # Step 4: Feature adjustments
    df["p_feat_raw"] = np.nan
    for race_id, g in df.groupby("RaceID"):
        idx = g.index
        p_vals = g["p_true"].values
        adj_unnorm = apply_feature_adjustments(p_vals, g, betas)
        adj_norm = normalize(adj_unnorm)
        df.loc[idx, "p_feat_raw"] = adj_norm

    # Step 5: Blend with market
    df["p_blended"] = np.nan
    for race_id, g in df.groupby("RaceID"):
        idx = g.index
        pm = g["p_true"].values
        pmode = g["p_feat_raw"].values
        blended = normalize(w_market * pm + (1 - w_market) * pmode)
        df.loc[idx, "p_blended"] = blended

    # Step 6: Favorite-longshot skew
    df["p_skewed"] = np.nan
    for race_id, g in df.groupby("RaceID"):
        idx = g.index
        p = g["p_blended"].values.astype(float)
        p2 = normalize(np.power(p, alpha))
        df.loc[idx, "p_skewed"] = p2

    # Step 7: Apply overround → final probability
    df["p_final"] = df["p_skewed"].astype(float) * df["race_overround"].astype(float)

    # Step 8: Convert to integer AdjustedPayoutOdds
    df["AdjustedPayoutOdds"] = round_to_step((100.0 / df["p_final"]), round_step).astype(int)

    # Step 9: Round probability columns for readability
    prob_cols = ["p_market_raw", "p_true", "p_feat_raw", "p_blended", "p_skewed", "p_final"]
    for c in prob_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(decimals)

    # Step 10: Output columns
    out_cols = [
        "RaceID", "Horse", "PortLouisOdds", "AdjustedPayoutOdds",
        "p_final", "p_true", "p_feat_raw"
    ]
    df[out_cols].to_csv(output_csv, index=False, float_format=f"%.{decimals}f")
    print(f"Adjusted odds CSV generated: {output_csv}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust horse racing odds using features and overround.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV with horse data")
    parser.add_argument("--output", "-o", required=True, help="Output CSV filename")
    parser.add_argument("--w_market", type=float, default=DEFAULT_W_MARKET, help="Blend weight for market")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Favorite-longshot skew exponent")
    parser.add_argument("--shrink", type=float, default=DEFAULT_SHRINK, help="Shrink factor for betas")
    parser.add_argument("--round_step", type=int, default=DEFAULT_ROUND_STEP, help="Round odds to nearest step")
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS, help="Decimals for probabilities")
    args = parser.parse_args()

    main(args.input, args.output,
         w_market=args.w_market,
         alpha=args.alpha,
         shrink=args.shrink,
         round_step=args.round_step,
         decimals=args.decimals)
