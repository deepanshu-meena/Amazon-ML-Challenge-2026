"""
Score a matching_results.tsv-style file against a ground truth file, using
the same macro F_0.5 formula as the leaderboard. Use this on a held-out
split of the TRAIN data (with known labels) before you ever touch the test
leaderboard, so you're not "spending" submissions to find out your score.

Usage:
    python -m src.evaluate --pred output/matching_results.tsv --truth dataset/train/train_ground_truth.tsv
"""
import argparse

from src.data_io import load_ground_truth
from src.model_utils import f_beta_macro


def load_pred(path):
    return load_ground_truth(path)  # same two-column format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    args = ap.parse_args()

    pred = load_pred(args.pred)
    truth = load_ground_truth(args.truth)

    score = f_beta_macro(pred, truth, beta=0.5)
    print(f"macro F_0.5 = {score:.4f}  (n_entities={len(truth)})")


if __name__ == "__main__":
    main()
