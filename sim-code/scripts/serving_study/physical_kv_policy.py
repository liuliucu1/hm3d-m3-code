"""Causal next-call estimates for offline KV retention experiments.

All times are seconds on the same monotonic clock as the KV lookup trace.
An update becomes visible only at its timestamp. A physical estimate is never
inferred from a completed trajectory or from a future request.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProgressUpdate:
    time_s: float
    episode_id: str
    role: str
    phase: str
    remaining_path_m: float | None = None
    linear_speed_mps: float | None = None
    remaining_turn_deg: float | None = None
    angular_speed_dps: float | None = None
    remaining_chunks: float | None = None
    chunk_rate_hz: float | None = None
    criticality: float | None = None

    @classmethod
    def from_record(cls, row: dict) -> "ProgressUpdate":
        fields = {name: row[name] for name in cls.__dataclass_fields__ if name in row}
        update = cls(**fields)
        if not math.isfinite(update.time_s):
            raise ValueError("progress time_s must be finite")
        if update.phase not in {"navigation", "manipulation", "waiting", "done"}:
            raise ValueError(f"unsupported progress phase: {update.phase}")
        for name in ("remaining_path_m", "remaining_turn_deg", "remaining_chunks"):
            value = getattr(update, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("linear_speed_mps", "angular_speed_dps", "chunk_rate_hz", "criticality"):
            value = getattr(update, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        return update

    def remaining_seconds(self) -> float | None:
        if self.phase == "navigation" and self.remaining_path_m is not None and self.linear_speed_mps:
            seconds = self.remaining_path_m / self.linear_speed_mps
            if self.remaining_turn_deg is not None and self.angular_speed_dps:
                seconds += self.remaining_turn_deg / self.angular_speed_dps
            return seconds
        if self.phase == "manipulation" and self.remaining_chunks is not None and self.chunk_rate_hz:
            return self.remaining_chunks / self.chunk_rate_hz
        return None


class NextCallPredictor:
    def __init__(self, default_gap_s: float = 5.0, ema_alpha: float = 0.5):
        if default_gap_s <= 0 or not 0 < ema_alpha <= 1:
            raise ValueError("invalid predictor parameters")
        self.default_gap_s = default_gap_s
        self.ema_alpha = ema_alpha
        self.last_call: dict[tuple[str, str], float] = {}
        self.gap: dict[tuple[str, str], float] = {}
        self.progress: dict[tuple[str, str], ProgressUpdate] = {}
        self.done_episodes: set[str] = set()

    def observe_call(self, session: tuple[str, str], time_s: float) -> None:
        old = self.last_call.get(session)
        if old is not None:
            if time_s < old:
                raise ValueError("calls must be time ordered")
            gap = time_s - old
            previous = self.gap.get(session)
            self.gap[session] = gap if previous is None else self.ema_alpha * gap + (1 - self.ema_alpha) * previous
        self.last_call[session] = time_s
        # A physical update describes the action before the call. It must not
        # predict another return after that call until a fresh update arrives.
        self.progress.pop(session, None)

    def observe_progress(self, update: ProgressUpdate) -> None:
        session = (update.episode_id, update.role)
        old = self.progress.get(session)
        if old is not None and update.time_s < old.time_s:
            raise ValueError("progress updates must be time ordered per session")
        if update.phase == "done":
            self.done_episodes.add(update.episode_id)
        else:
            if update.episode_id in self.done_episodes:
                raise ValueError("progress after episode completion")
            self.progress[session] = update

    def estimate(self, session: tuple[str, str], now_s: float, use_progress: bool) -> tuple[float, float, bool]:
        if session[0] in self.done_episodes:
            return math.inf, 1.0, True
        update = self.progress.get(session) if use_progress else None
        if update is not None and update.time_s > now_s:
            raise ValueError("future progress update is not observable")
        residual = update.remaining_seconds() if update is not None else None
        if residual is not None and now_s <= update.time_s + residual + 2.0:
            eta = max(0.0, update.time_s + residual - now_s)
        else:
            last = self.last_call.get(session, now_s)
            predicted = last + self.gap.get(session, self.default_gap_s)
            # An interval estimate that is already overdue is stale, not a
            # promise that this session will call immediately. Age it out.
            eta = predicted - now_s if predicted >= now_s else self.default_gap_s + 10 * (now_s - predicted)
        default_weight = 4.0 if session[1] == "target_review" else 1.0
        weight = update.criticality if update is not None and update.criticality is not None else default_weight
        return eta, weight, False
