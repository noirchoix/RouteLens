from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from typing import Any, Protocol

from reacts.chemistry.mapping import AtomMappingEngine, MappingResult, validate_mapped_reaction
from reacts.chemistry.reactions import canonicalize_reaction, parse_reaction
from reacts.contracts import MappingStatus


class MappingBackend(Protocol):
    name: str

    def map_batch(self, reactions: list[str]) -> list[MappingResult]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class RXNTokenEligibility:
    token_count: int | None
    token_limit: int
    eligible: bool
    count_method: str



def _result_to_payload(result: MappingResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "mapped_reaction_smiles": result.mapped_reaction_smiles,
        "backend": result.backend,
        "confidence": float(result.confidence),
        "atom_coverage": float(result.atom_coverage),
        "diagnostics": list(result.diagnostics),
        "error_code": result.error_code,
        "fallback_status": result.fallback_status,
    }


def _payload_to_result(payload: dict[str, Any]) -> MappingResult:
    return MappingResult(
        status=MappingStatus(str(payload["status"])),
        mapped_reaction_smiles=payload.get("mapped_reaction_smiles"),
        backend=str(payload.get("backend") or "mcs_fallback"),
        confidence=float(payload.get("confidence") or 0.0),
        atom_coverage=float(payload.get("atom_coverage") or 0.0),
        diagnostics=tuple(str(item) for item in payload.get("diagnostics") or ()),
        error_code=str(payload["error_code"]) if payload.get("error_code") else None,
        fallback_status=str(payload["fallback_status"]) if payload.get("fallback_status") else None,
    )


def _bounded_mcs_worker(
    connection: Connection,
    reaction: str,
    min_coverage: float,
    rdkit_timeout_seconds: int,
) -> None:
    """Run one MCS mapping in a killable child process.

    This function is module-level so the Windows ``spawn`` multiprocessing
    context can import it. Only JSON-compatible payloads cross the pipe.
    """

    try:
        engine = AtomMappingEngine(
            "mcs_fallback",
            min_coverage=min_coverage,
            timeout_seconds=rdkit_timeout_seconds,
        )
        result = engine.map_reaction(reaction)
        if result.status == MappingStatus.MAPPED:
            fallback_status = "mapped"
            error_code = None
        elif result.status == MappingStatus.LOW_CONFIDENCE:
            fallback_status = "low_confidence"
            error_code = "mcs_low_confidence"
        elif result.status == MappingStatus.NOT_ELIGIBLE:
            fallback_status = "failed"
            error_code = "mcs_not_eligible"
        else:
            fallback_status = "failed"
            error_code = result.error_code or "mcs_failed"
        connection.send(
            _result_to_payload(
                replace(
                    result,
                    error_code=error_code,
                    fallback_status=fallback_status,
                )
            )
        )
    except BaseException as exc:  # child boundary: always return a deterministic failure payload
        connection.send(
            _result_to_payload(
                MappingResult(
                    MappingStatus.FAILED,
                    None,
                    "mcs_fallback",
                    0.0,
                    0.0,
                    (f"{type(exc).__name__}: {exc}",),
                    error_code="mcs_worker_error",
                    fallback_status="failed",
                )
            )
        )
    finally:
        connection.close()


@dataclass
class RXNMapperBackend:
    mapper: object
    min_confidence: float = 0.50
    min_coverage: float = 0.60
    max_token_length: int = 512
    name: str = "rxnmapper"

    def _tokenizer(self) -> object | None:
        for attribute in ("tokenizer", "_tokenizer", "smiles_tokenizer"):
            tokenizer = getattr(self.mapper, attribute, None)
            if tokenizer is not None:
                return tokenizer
        return None

    def _model_token_limit(self) -> int:
        candidates: list[int] = [int(self.max_token_length)]
        tokenizer = self._tokenizer()
        value = getattr(tokenizer, "model_max_length", None) if tokenizer is not None else None
        if isinstance(value, int) and 8 <= value <= 100_000:
            candidates.append(value)
        model = getattr(self.mapper, "model", None) or getattr(self.mapper, "_model", None)
        config = getattr(model, "config", None)
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and 8 <= value <= 100_000:
            candidates.append(value)
        return min(candidates)

    def token_eligibility(self, canonical_reaction: str) -> RXNTokenEligibility:
        limit = self._model_token_limit()
        tokenizer = self._tokenizer()
        if tokenizer is not None:
            tokenize = getattr(tokenizer, "tokenize", None)
            if callable(tokenize):
                tokens = tokenize(canonical_reaction)
                special_tokens = 2
                special_counter = getattr(tokenizer, "num_special_tokens_to_add", None)
                if callable(special_counter):
                    try:
                        special_tokens = int(special_counter(pair=False))
                    except TypeError:
                        special_tokens = int(special_counter(False))
                    except Exception:
                        special_tokens = 2
                count = len(tokens) + max(special_tokens, 0)
                return RXNTokenEligibility(count, limit, count <= limit, "tokenizer.tokenize")

            encode = getattr(tokenizer, "encode", None)
            if callable(encode):
                try:
                    encoded = encode(canonical_reaction, add_special_tokens=True, truncation=False)
                except TypeError:
                    encoded = encode(canonical_reaction, add_special_tokens=True)
                count = len(encoded)
                return RXNTokenEligibility(count, limit, count <= limit, "tokenizer.encode")

        # This is a conservative safety fallback only. RXNMapper 0.4.3 exposes
        # a tokenizer in supported environments; character count is an upper
        # bound that prevents a missing tokenizer attribute from disabling the
        # guard for obviously oversized reactions.
        count = len(canonical_reaction)
        return RXNTokenEligibility(count, limit, count <= limit, "character_upper_bound")

    def _map_recursive(self, reactions: list[str]) -> list[MappingResult]:
        if not reactions:
            return []
        try:
            raw = self.mapper.get_attention_guided_atom_maps(reactions)
            if len(raw) != len(reactions):
                raise RuntimeError(f"RXNMapper returned {len(raw)} rows for {len(reactions)} inputs")
            output: list[MappingResult] = []
            for item in raw:
                mapped = str(item.get("mapped_rxn") or "")
                confidence = float(item.get("confidence") or 0.0)
                valid, coverage, issues = validate_mapped_reaction(mapped)
                status = (
                    MappingStatus.MAPPED
                    if valid and confidence >= self.min_confidence and coverage >= self.min_coverage
                    else MappingStatus.LOW_CONFIDENCE
                )
                output.append(
                    MappingResult(
                        status=status,
                        mapped_reaction_smiles=mapped or None,
                        backend=self.name,
                        confidence=confidence,
                        atom_coverage=coverage,
                        diagnostics=tuple(issues),
                        error_code=None if status == MappingStatus.MAPPED else "rxnmapper_low_confidence",
                        rxnmapper_eligible=True,
                    )
                )
            return output
        except Exception as exc:
            if len(reactions) == 1:
                return [
                    MappingResult(
                        MappingStatus.FAILED,
                        None,
                        self.name,
                        0.0,
                        0.0,
                        (f"{type(exc).__name__}: {exc}",),
                        error_code="rxnmapper_error",
                        rxnmapper_eligible=True,
                    )
                ]
            midpoint = len(reactions) // 2
            return self._map_recursive(reactions[:midpoint]) + self._map_recursive(reactions[midpoint:])

    def map_batch(self, reactions: list[str]) -> list[MappingResult]:
        canonical: list[str] = []
        positions: list[int] = []
        token_metadata: dict[int, RXNTokenEligibility] = {}
        output: list[MappingResult | None] = [None] * len(reactions)
        for index, reaction in enumerate(reactions):
            parsed = parse_reaction(reaction)
            normalized = canonicalize_reaction(reaction)
            if not parsed.parse_ok or normalized is None:
                output[index] = MappingResult(
                    MappingStatus.NOT_ELIGIBLE,
                    None,
                    self.name,
                    0.0,
                    0.0,
                    (parsed.failure_class.value,),
                    error_code="rxnmapper_input_not_eligible",
                    rxnmapper_eligible=False,
                )
                continue

            eligibility = self.token_eligibility(normalized)
            token_metadata[index] = eligibility
            if not eligibility.eligible:
                output[index] = MappingResult(
                    MappingStatus.FAILED,
                    None,
                    self.name,
                    0.0,
                    0.0,
                    (
                        "rxnmapper_sequence_too_long",
                        f"token_count={eligibility.token_count}",
                        f"token_limit={eligibility.token_limit}",
                        f"count_method={eligibility.count_method}",
                    ),
                    error_code="rxnmapper_sequence_too_long",
                    rxnmapper_token_count=eligibility.token_count,
                    rxnmapper_token_limit=eligibility.token_limit,
                    rxnmapper_eligible=False,
                )
                continue

            canonical.append(normalized)
            positions.append(index)

        mapped = self._map_recursive(canonical)
        for position, result in zip(positions, mapped):
            eligibility = token_metadata[position]
            output[position] = replace(
                result,
                rxnmapper_token_count=eligibility.token_count,
                rxnmapper_token_limit=eligibility.token_limit,
                rxnmapper_eligible=True,
            )
        return [item for item in output if item is not None]

    def close(self) -> None:
        self.mapper = None


@dataclass
class MCSBackend:
    min_coverage: float = 0.60
    timeout_seconds: int = 3
    process_timeout_seconds: int = 30
    name: str = "mcs_fallback"
    worker_target: Any = field(default=_bounded_mcs_worker, repr=False, compare=False)

    def _map_one_bounded(self, reaction: str) -> MappingResult:
        context = mp.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=self.worker_target,
            args=(child, reaction, self.min_coverage, self.timeout_seconds),
            daemon=True,
        )
        started = time.perf_counter()
        try:
            process.start()
            child.close()
            process.join(timeout=max(float(self.process_timeout_seconds), 0.1))
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
                if process.is_alive() and hasattr(process, "kill"):
                    process.kill()
                    process.join(timeout=2.0)
                elapsed = time.perf_counter() - started
                return MappingResult(
                    MappingStatus.FAILED,
                    None,
                    self.name,
                    0.0,
                    0.0,
                    (
                        "mcs_timeout",
                        f"process_timeout_seconds={self.process_timeout_seconds}",
                        f"elapsed_seconds={elapsed:.3f}",
                    ),
                    error_code="mcs_timeout",
                    fallback_status="timeout",
                )
            if parent.poll(1.0):
                return _payload_to_result(parent.recv())
            return MappingResult(
                MappingStatus.FAILED,
                None,
                self.name,
                0.0,
                0.0,
                (f"mcs_worker_exitcode={process.exitcode}", "mcs_worker_returned_no_result"),
                error_code="mcs_worker_no_result",
                fallback_status="failed",
            )
        except BaseException as exc:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            return MappingResult(
                MappingStatus.FAILED,
                None,
                self.name,
                0.0,
                0.0,
                (f"{type(exc).__name__}: {exc}",),
                error_code="mcs_process_error",
                fallback_status="failed",
            )
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

    def map_batch(self, reactions: list[str]) -> list[MappingResult]:
        return [self._map_one_bounded(reaction) for reaction in reactions]

    def close(self) -> None:
        return None
