"""Helpers for preserving physical durations when simulation tick size changes."""

from __future__ import annotations

import copy
import math
from typing import Any


LEGACY_RECIPE_TICK_MINUTES = 5.0


def ticks_for_minutes(minutes: float, tick_minutes: float) -> int:
    return max(1, int(math.ceil(float(minutes) / float(tick_minutes))))


def scale_tick_count(
    ticks: float, target_tick_minutes: float,
    source_tick_minutes: float = LEGACY_RECIPE_TICK_MINUTES,
) -> int:
    physical_minutes = float(ticks) * float(source_tick_minutes)
    return ticks_for_minutes(physical_minutes, target_tick_minutes)


def normalize_tick_based_config(
    value: Any,
    target_tick_minutes: float,
    source_tick_minutes: float = LEGACY_RECIPE_TICK_MINUTES,
) -> Any:
    """Deep-copy legacy recipe data while preserving physical time/rates.

    Recipe quantities and BOMs are untouched.  Only explicitly tick-denominated
    durations/intervals and continuous per-tick rates are converted.
    """
    ratio = float(target_tick_minutes) / float(source_tick_minutes)

    def convert(node: Any) -> Any:
        if isinstance(node, dict):
            result = {}
            for key, item in node.items():
                if (
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                ):
                    if key.endswith("_ticks"):
                        result[key] = scale_tick_count(
                            item, target_tick_minutes, source_tick_minutes
                        )
                        continue
                    if key.endswith("_per_tick"):
                        result[key] = float(item) * ratio
                        continue
                    if "_per_100_ticks" in key:
                        result[key] = float(item) * ratio
                        continue
                result[key] = convert(item)
            return result
        if isinstance(node, list):
            return [convert(item) for item in node]
        return copy.deepcopy(node)

    return convert(value)
