"""Compare causal KV eviction policies on a frozen vLLM block trace.

This is an allocation-only replay: it models contiguous-prefix hits, not
in-flight pinning, transfer time, prefetch, or robot waiting time. Progress
updates are optional; without them ``progress`` reduces to ``interval``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from physical_kv_policy import NextCallPredictor, ProgressUpdate


def load_requests(analysis_dir: Path) -> list[dict]:
    replay = {}
    with (analysis_dir / "robothor_k8_kvprobe_replay.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            replay[row["request_id"]] = row
    stored = defaultdict(list)
    events = []
    with (analysis_dir / "kv_probe_robothor_k8.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["kind"] == "stored":
                stored[event["request_id"]].append(event["hash"])
            elif event["kind"] == "lookup":
                events.append(event)
    requests = []
    for event in events:
        request_id = event["request_id"].removeprefix("chatcmpl-").rsplit("-", 1)[0]
        row = replay[request_id]
        look = list(event["hashes"])
        seen = set(look)
        extra = []
        for block in stored[event["request_id"]]:
            if block not in seen:
                extra.append(block)
                seen.add(block)
        requests.append({
            "time_s": event["time_ns"] / 1e9,
            "look": look,
            "all": look + extra,
            "session": (row["episode_id"], row["role"]),
            "role": row["role"],
        })
    if not requests:
        raise ValueError("no KV lookup events")
    if any(a["time_s"] > b["time_s"] for a, b in zip(requests, requests[1:])):
        raise ValueError("KV lookups are not time ordered")
    return requests


def load_updates(path: Path | None) -> list[ProgressUpdate]:
    if path is None:
        return []
    with path.open() as stream:
        updates = [ProgressUpdate.from_record(json.loads(line)) for line in stream if line.strip()]
    updates.sort(key=lambda update: update.time_s)
    return updates


def replay(requests: list[dict], capacity: int, policy: str,
           updates: list[ProgressUpdate] | None = None) -> dict:
    if capacity <= 0 or policy not in {"lru", "interval", "progress"}:
        raise ValueError("invalid capacity or policy")
    updates = updates or []
    predictor = NextCallPredictor()
    resident: dict[str, tuple[int, int]] = {}  # block -> (last use, negative position)
    owners: dict[str, set[tuple[str, str]]] = defaultdict(set)
    misses, lookups = Counter(), Counter()
    update_index = 0
    used_updates = 0
    for sequence, request in enumerate(requests):
        now = request["time_s"]
        while update_index < len(updates) and updates[update_index].time_s <= now:
            predictor.observe_progress(updates[update_index])
            update_index += 1
            used_updates += 1
        session = request["session"]
        hit = 0
        for block in request["look"]:
            if block not in resident:
                break
            hit += 1
        role = request["role"]
        lookups[role] += len(request["look"])
        misses[role] += len(request["look"]) - hit
        predictor.observe_call(session, now)
        # A shared block is useful when *any* living owner is about to return.
        for position, block in enumerate(request["all"]):
            owners[block].add(session)
            resident[block] = (sequence, -position)
        excess = len(resident) - capacity
        if excess > 0:
            protected = set(request["all"])
            candidates = [block for block in resident if block not in protected]
            if policy == "lru":
                candidates.sort(key=lambda block: (resident[block], block))
            else:
                session_owners = {owner for block in candidates for owner in owners[block]}
                estimates = {owner: predictor.estimate(owner, now, policy == "progress")
                             for owner in session_owners}
                def eviction_key(block: str) -> tuple[float, int, int, str]:
                    predictions = [estimates[owner] for owner in owners[block]]
                    living = [eta / weight for eta, weight, done in predictions if not done]
                    # Larger score is evicted first. Finished episodes go first.
                    score = min(living) if living else float("inf")
                    last_use, negative_position = resident[block]
                    return score, -last_use, -negative_position, block
                candidates.sort(key=eviction_key, reverse=True)
            victims = candidates[:excess]
            if len(victims) < excess:
                # A single request exceeds capacity; discard its tail first.
                victims.extend(reversed(request["all"][capacity:]))
            for victim in victims:
                del resident[victim]
    return {
        "capacity_blocks": capacity, "policy": policy,
        "lookups": dict(lookups), "misses": dict(misses),
        "total_misses": sum(misses.values()),
        "progress_updates_used": used_updates if policy == "progress" else 0,
        "progress_updates_total": len(updates) if policy == "progress" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    parser.add_argument("--capacities", default="3146,2200,1400,1000")
    parser.add_argument("--policies", default="lru,interval,progress")
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    requests = load_requests(args.analysis_dir)
    updates = load_updates(args.progress_jsonl)
    results = [replay(requests, int(capacity), policy, updates)
               for capacity in args.capacities.split(",")
               for policy in args.policies.split(",")]
    document = {"trace": str(args.analysis_dir), "progress_source":
                str(args.progress_jsonl) if args.progress_jsonl else None,
                "limitations": "No in-flight pinning, time feedback, prefetch, or wait-time model.",
                "results": results}
    output = json.dumps(document, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
