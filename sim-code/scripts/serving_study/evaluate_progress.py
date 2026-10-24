"""Evaluate causal next-call predictions against later calls as labels only."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path

from kv_progress_replay import load_requests, load_updates
from physical_kv_policy import NextCallPredictor


def evaluate(requests: list[dict], updates: list) -> dict:
    calls = defaultdict(list)
    for request in requests:
        calls[request["session"]].append(request["time_s"])
    predictor = NextCallPredictor()
    next_request = 0
    errors = {"interval": [], "progress": []}
    pair_correct = {"interval": 0, "progress": 0}
    pairs = 0
    evaluated = 0
    for update in updates:
        now = update.time_s
        while next_request < len(requests) and requests[next_request]["time_s"] <= now:
            request = requests[next_request]
            predictor.observe_call(request["session"], request["time_s"])
            next_request += 1
        predictor.observe_progress(update)
        if update.phase == "done":
            continue
        session = (update.episode_id, update.role)
        future = calls.get(session, [])
        index = bisect_right(future, now)
        if index >= len(future):
            continue  # right censored: no observed return, not a zero-second label
        truth = future[index] - now
        for policy in errors:
            estimate = predictor.estimate(session, now, policy == "progress")[0]
            errors[policy].append(abs(estimate - truth))
        evaluated += 1
        # At this actual decision instant, compare all observable sessions with
        # a later call. The future is used solely to score the ordering.
        for other, other_calls in calls.items():
            if other == session or other not in predictor.last_call:
                continue
            other_index = bisect_right(other_calls, now)
            if other_index >= len(other_calls):
                continue
            other_truth = other_calls[other_index] - now
            if other_truth == truth:
                continue
            pairs += 1
            for policy in pair_correct:
                left = predictor.estimate(session, now, policy == "progress")[0]
                right = predictor.estimate(other, now, policy == "progress")[0]
                pair_correct[policy] += (left - right) * (truth - other_truth) > 0
    return {
        "updates_total": len(updates), "updates_evaluated": evaluated,
        "pairwise_comparisons": pairs,
        "mean_absolute_error_s": {policy: sum(values) / len(values) if values else None
                                  for policy, values in errors.items()},
        "pairwise_accuracy": {policy: pair_correct[policy] / pairs if pairs else None
                              for policy in pair_correct},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("progress_jsonl", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate(load_requests(args.analysis_dir), load_updates(args.progress_jsonl))
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
