from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from reacts.data.parsing import parse_list


@dataclass(frozen=True)
class ConditionValidity:
    temperature_observed_c: float | None
    temperature_clean_c: float | None
    temperature_valid: bool | None
    time_observed_h: float | None
    time_clean_h: float | None
    time_valid: bool | None
    status: str
    issues: tuple[str, ...]


def temperature_bucket(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value < 0:
        return "<0"
    if value <= 25:
        return "0-25"
    if value <= 60:
        return "25-60"
    if value <= 100:
        return "60-100"
    return "100+"


def time_bucket(value: float | None) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if value < 1:
        return "<1h"
    if value <= 4:
        return "1-4h"
    if value <= 16:
        return "4-16h"
    if value <= 24:
        return "16-24h"
    return "24h+"


def _number(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def validate_conditions(
    temperature_c: object,
    time_h: object,
    *,
    temperature_min_c: float = -150.0,
    temperature_max_c: float = 350.0,
    time_min_h: float = 1.0 / 3600.0,
    time_max_h: float = 8760.0,
) -> ConditionValidity:
    temp = _number(temperature_c)
    duration = _number(time_h)
    issues: list[str] = []

    temp_valid: bool | None = None if temp is None else temperature_min_c <= temp <= temperature_max_c
    time_valid: bool | None = None if duration is None else time_min_h <= duration <= time_max_h
    if temp_valid is False:
        issues.append("temperature_out_of_plausibility_range")
    if time_valid is False:
        issues.append("time_out_of_plausibility_range")

    if issues:
        status = "suspicious"
    elif temp is None and duration is None:
        status = "missing"
    elif temp is None or duration is None:
        status = "partial"
    else:
        status = "valid"

    return ConditionValidity(
        temperature_observed_c=temp,
        temperature_clean_c=temp if temp_valid else None,
        temperature_valid=temp_valid,
        time_observed_h=duration,
        time_clean_h=duration if time_valid else None,
        time_valid=time_valid,
        status=status,
        issues=tuple(issues),
    )


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        token = str(value).strip()
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output


def normalize_condition_lists(solvents: object, agents: object) -> tuple[list[str], list[str]]:
    return unique_preserving_order(parse_list(solvents)), unique_preserving_order(parse_list(agents))

@dataclass(frozen=True)
class MiddleConditionParse:
    temperature_c: float | None
    time_h: float | None
    confidence: str
    method: str
    issues: tuple[str, ...]
    numeric_tokens: tuple[float, ...]


def reparse_multistep_middle(reaction_smiles: str) -> MiddleConditionParse:
    """Recover temperature/time from the legacy `reactants>middle>products` encoding.

    The source-generating code concatenated condition families in the practical
    order chemicals, temperature, time. The legacy normalizer then checked every
    bare numeric as time first, which converted temperatures such as 100 C to
    100 seconds. This conservative repair only promotes numerics when positional
    and magnitude evidence agree; ambiguous values remain flagged.
    """
    text = str(reaction_smiles or "").strip()
    if ">>" in text or text.count(">") < 2:
        return MiddleConditionParse(None, None, "none", "not_three_field", (), ())
    _, middle, _ = text.split(">", 2)
    raw_tokens = [token.strip() for token in middle.split(".") if token.strip()]
    if not raw_tokens:
        return MiddleConditionParse(None, None, "none", "empty_middle", (), ())

    temp_aliases = {"room temperature": 25.0, "rt": 25.0, "ambient": 25.0, "ambient temperature": 25.0, "ice bath": 0.0}
    time_aliases = {"overnight": 16.0}
    numeric: list[float] = []
    temperature: float | None = None
    duration: float | None = None
    issues: list[str] = []

    for token in raw_tokens:
        lowered = token.lower()
        if lowered in temp_aliases:
            temperature = temp_aliases[lowered]
            continue
        if lowered in time_aliases:
            duration = time_aliases[lowered]
            continue
        try:
            numeric.append(float(token))
        except ValueError:
            continue

    if not numeric:
        confidence = "high" if temperature is not None or duration is not None else "none"
        return MiddleConditionParse(temperature, duration, confidence, "alias_only", tuple(issues), ())

    unresolved = list(numeric)
    # Negative values are chemically plausible temperatures and impossible durations.
    negatives = [value for value in unresolved if value < 0]
    if temperature is None and negatives:
        temperature = negatives[0]
        unresolved.remove(temperature)

    # In the source encoding, a large final numeric is normally duration in seconds.
    if duration is None and unresolved:
        last = unresolved[-1]
        if last > 300 or (len(unresolved) >= 2 and last >= 60):
            duration = last / 3600.0
            unresolved = unresolved[:-1]

    # Remaining plausible numerics precede time and therefore represent temperature.
    plausible_temps = [value for value in unresolved if -150 <= value <= 350]
    if temperature is None and plausible_temps:
        temperature = plausible_temps[0]
        if len(plausible_temps) > 1:
            issues.append("multi_temperature_profile_primary_selected")

    if duration is None and len(numeric) == 1 and numeric[0] > 350:
        duration = numeric[0] / 3600.0
    elif duration is None and len(numeric) == 1 and -150 <= numeric[0] <= 350:
        issues.append("single_numeric_assumed_temperature")

    if temperature is not None and duration is not None:
        confidence = "high"
    elif temperature is not None or duration is not None:
        confidence = "medium" if not issues else "low"
    else:
        confidence = "none"
    return MiddleConditionParse(temperature, duration, confidence, "positional_numeric_repair", tuple(issues), tuple(numeric))
