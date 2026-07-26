from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import uvicorn

from reacts.data.source import ArtifactSource
from reacts.ml.specialists import MULTILABEL_TASKS, REGRESSION_TASKS
from reacts.ml.tasks import TASKS
from reacts.publishing.huggingface import build_huggingface_bundle
from reacts.services.application import Application
from reacts.settings import Settings


def _settings(args: argparse.Namespace) -> Settings:
    cfg = Settings(project_root=Path(args.project_root)).resolve()
    if getattr(args, "source", None):
        cfg.source_artifact = Path(args.source).resolve()
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reacts", description="REACTS Product Two CLI")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-source", help="Inventory and audit the legacy artifact")
    audit.add_argument("--source")

    canonical = sub.add_parser("build-canonical", help="Build Product One canonical routes and steps")
    canonical.add_argument("--source")
    canonical.add_argument("--csv-fallback", action="store_true")

    train = sub.add_parser("train", help="Train classification tasks with task-specific release gates")
    train.add_argument("--tasks", nargs="+", default=["parse_failure_class", "reaction_family"])
    train.add_argument("--dataset-version", default="uspto_multistep_contextual_v2")
    train.add_argument("--max-rows", type=int)
    train.add_argument("--request-promotion", action="store_true")

    specialist = sub.add_parser("train-specialist", help="Train Product Two multilabel or interval-regression tasks")
    specialist.add_argument("--tasks", nargs="+", required=True, choices=sorted([*MULTILABEL_TASKS, *REGRESSION_TASKS]))
    specialist.add_argument("--max-rows", type=int)
    specialist.add_argument("--request-promotion", action="store_true")

    index = sub.add_parser("build-index", help="Build Product One reaction index")
    index.add_argument("--max-rows", type=int)

    product_one = sub.add_parser("product-one", help="Compatibility command for the Product One baseline pipeline")
    product_one.add_argument("--source")
    product_one.add_argument("--tasks", nargs="+", default=["parse_validity", "primary_solvent", "time_bucket", "temperature_bucket", "agent_presence"])
    product_one.add_argument("--max-rows", type=int)
    product_one.add_argument("--csv-fallback", action="store_true")
    product_one.add_argument("--request-promotion", action="store_true")

    freeze = sub.add_parser("freeze-product-one", help="Freeze and reclassify Product One as v1.0.0-baseline")

    for command_name in ["build-contextual-v2", "build-contextual"]:
        contextual = sub.add_parser(
            command_name,
            help="Build the resumable context-only canonical v2 dataset from stored Product One artifacts",
        )
        contextual.add_argument("--csv-fallback", action="store_true")
        contextual.add_argument("--resume", action="store_true")
        contextual.add_argument("--clean", action="store_true")

    benchmark = sub.add_parser("benchmark-mapper", help="Benchmark bounded atom-mapping batch sizes before a full run")
    benchmark.add_argument("--backend", default="rxnmapper", choices=["rxnmapper", "mcs", "mcs_fallback"])
    benchmark.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    benchmark.add_argument("--sample-size", type=int, default=512)

    map_reactions = sub.add_parser("map-reactions", help="Run persistent, batched and resumable atom mapping")
    map_reactions.add_argument("--backend", default="rxnmapper", choices=["auto", "rxnmapper", "mcs", "mcs_fallback"])
    map_reactions.add_argument("--fallback-backend", default="mcs", choices=["mcs", "mcs_fallback", "none"])
    map_reactions.add_argument("--allow-auto-fallback", action="store_true")
    map_reactions.add_argument("--batch-size", type=int, default=16)
    map_reactions.add_argument("--workers", type=int, default=1)
    map_reactions.add_argument("--prefetch-batches", type=int, default=2)
    map_reactions.add_argument("--shard-size", type=int, default=5000)
    map_reactions.add_argument("--rxnmapper-token-limit", type=int, default=512)
    map_reactions.add_argument("--fallback-timeout-seconds", type=int, default=30)
    map_reactions.add_argument("--resume", action="store_true")
    map_reactions.add_argument("--max-rows", type=int)
    map_reactions.add_argument("--csv-fallback", action="store_true")

    derive = sub.add_parser("derive-reaction-centres", help="Derive bond changes and structural families independently")
    derive.add_argument("--resume", action="store_true")
    derive.add_argument("--include-mcs", action="store_true")
    derive.add_argument("--min-confidence", type=float, default=0.50)
    derive.add_argument("--max-rows", type=int)
    derive.add_argument("--csv-fallback", action="store_true")

    resplit = sub.add_parser(
        "rebuild-product-two-splits",
        help="Rebuild final Product Two splits using patent/reaction connected components without rerunning mapping",
    )
    resplit.add_argument("--seed", type=int, default=42)
    resplit.add_argument("--csv-fallback", action="store_true")

    contextual_index = sub.add_parser("build-product-two-indexes", help="Build Product Two evidence indexes after split governance")
    contextual_index.add_argument("--max-rows", type=int)
    contextual_index.add_argument("--resume", action="store_true")
    legacy_contextual_index = sub.add_parser("build-contextual-index", help="Compatibility alias for build-product-two-indexes")
    legacy_contextual_index.add_argument("--max-rows", type=int)

    train_two = sub.add_parser("train-product-two", help="Train Product Two models only after structural qualification")
    train_two.add_argument("--max-rows", type=int)
    train_two.add_argument("--classification-tasks", nargs="+", default=["parse_failure_class", "reaction_family"])
    train_two.add_argument(
        "--specialist-tasks",
        nargs="+",
        default=[
            "solvent_multilabel", "solvent_family_multilabel", "time_regression",
            "temperature_regression", "agent_family_multilabel", "catalyst_family_multilabel",
        ],
    )
    train_two.add_argument("--request-promotion", action="store_true")

    anomaly = sub.add_parser("build-anomaly-model", help="Fit transparent family-conditional condition anomaly statistics")

    validate = sub.add_parser("validate-product-two", help="Run strict leakage, loading, API and retrieval acceptance checks")

    lock = sub.add_parser("lock-product-two", help="Lock Product Two after strict scientific acceptance")
    lock.add_argument("--release-id", default="v2.0.12")

    product_two = sub.add_parser("product-two", help="Run the staged, resumable Product Two workflow")
    stages = ["context", "mapping", "derivation", "splits", "indexes", "training", "validation"]
    product_two.add_argument("--from-stage", choices=stages, default="context")
    product_two.add_argument("--stop-after", choices=stages, default="validation")
    product_two.add_argument("--resume", action="store_true")
    product_two.add_argument("--csv-fallback", action="store_true")
    product_two.add_argument("--mapping-backend", default="rxnmapper", choices=["auto", "rxnmapper", "mcs", "mcs_fallback"])
    product_two.add_argument("--fallback-backend", default="mcs", choices=["mcs", "mcs_fallback", "none"])
    product_two.add_argument("--allow-auto-fallback", action="store_true")
    product_two.add_argument("--batch-size", type=int, default=16)
    product_two.add_argument("--shard-size", type=int, default=5000)
    product_two.add_argument("--max-rows", type=int)
    product_two.add_argument("--include-mcs", action="store_true")
    product_two.add_argument("--classification-tasks", nargs="+", default=["parse_failure_class", "reaction_family"])
    product_two.add_argument(
        "--specialist-tasks", nargs="+",
        default=[
            "solvent_multilabel", "solvent_family_multilabel", "time_regression",
            "temperature_regression", "agent_family_multilabel", "catalyst_family_multilabel",
        ],
    )
    product_two.add_argument("--request-promotion", action="store_true")
    product_two.add_argument("--lock-release", action="store_true")
    product_two.add_argument("--release-id", default="v2.0.12")

    export_hf = sub.add_parser("export-hf", help="Create a publication-ready Hugging Face dataset bundle")
    export_hf.add_argument("--destination", required=True)
    export_hf.add_argument("--include-models", action="store_true")
    export_hf.add_argument("--contextual", action="store_true")

    serve = sub.add_parser("serve", help="Start the API and embedded UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", default=8000, type=int)
    serve.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = _settings(args)

    if args.command == "audit-source":
        source = ArtifactSource(cfg.source_artifact)
        print(json.dumps(source.inventory().__dict__, indent=2, default=str))
        return 0

    app = Application(
        cfg,
        read_only_registry=args.command == "validate-product-two",
    )
    if args.command == "build-canonical":
        print(json.dumps(app.build_canonical(prefer_parquet=not args.csv_fallback), indent=2, default=str))
        return 0
    if args.command == "train":
        unknown = [task for task in args.tasks if task not in TASKS]
        if unknown:
            parser.error(f"Unknown classification tasks: {unknown}")
        result = app.trainer(args.dataset_version, args.max_rows).train_many(
            args.tasks, promote_validated=args.request_promotion
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "train-specialist":
        trainer = app.specialist_trainer(args.max_rows)
        result = {
            task: trainer.train(task, request_promotion=args.request_promotion)
            for task in args.tasks
        }
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "build-index":
        print(json.dumps(app.build_index(args.max_rows), indent=2, default=str))
        return 0
    if args.command == "product-one":
        canonical = app.build_canonical(prefer_parquet=not args.csv_fallback)
        models = app.trainer(canonical["dataset_version"], args.max_rows).train_many(
            args.tasks, promote_validated=args.request_promotion
        )
        index_result = app.build_index(args.max_rows)
        report: dict[str, Any] = {"canonical": canonical, "models": models, "index": index_result}
        report_path = cfg.reports_dir / "product_one_run.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"report": report_path.relative_to(cfg.project_root).as_posix(), **report}, indent=2, default=str))
        return 0
    if args.command == "freeze-product-one":
        print(json.dumps(app.freeze_product_one(), indent=2, default=str))
        return 0
    if args.command in {"build-contextual-v2", "build-contextual"}:
        result = app.build_contextual_canonical(
            prefer_parquet=not args.csv_fallback,
            resume=args.resume,
            clean=args.clean,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "benchmark-mapper":
        result = app.benchmark_mapper(
            backend=args.backend,
            batch_sizes=tuple(args.batch_sizes),
            sample_size=args.sample_size,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "map-reactions":
        result = app.map_reactions(
            backend=args.backend,
            fallback_backend=None if args.fallback_backend == "none" else args.fallback_backend,
            allow_auto_fallback=args.allow_auto_fallback,
            batch_size=args.batch_size,
            workers=args.workers,
            prefetch_batches=args.prefetch_batches,
            shard_size=args.shard_size,
            rxnmapper_token_limit=args.rxnmapper_token_limit,
            fallback_process_timeout_seconds=args.fallback_timeout_seconds,
            resume=args.resume,
            max_rows=args.max_rows,
            prefer_parquet=not args.csv_fallback,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "derive-reaction-centres":
        result = app.derive_reaction_centres(
            resume=args.resume,
            include_mcs=args.include_mcs,
            min_confidence=args.min_confidence,
            max_rows=args.max_rows,
            prefer_parquet=not args.csv_fallback,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "rebuild-product-two-splits":
        result = app.rebuild_product_two_splits(
            prefer_parquet=not args.csv_fallback,
            random_seed=args.seed,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command in {"build-product-two-indexes", "build-contextual-index"}:
        print(json.dumps(app.build_contextual_index(args.max_rows), indent=2, default=str))
        return 0
    if args.command == "train-product-two":
        report: dict[str, Any] = {}
        report["classification_models"] = app.trainer(
            "uspto_multistep_contextual_v2", args.max_rows
        ).train_many(args.classification_tasks, promote_validated=args.request_promotion)
        specialist_trainer = app.specialist_trainer(args.max_rows)
        report["specialist_models"] = {
            task: specialist_trainer.train(task, request_promotion=args.request_promotion)
            for task in args.specialist_tasks
        }
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "build-anomaly-model":
        print(json.dumps(app.build_anomaly_model(), indent=2, default=str))
        return 0
    if args.command == "validate-product-two":
        result = app.validate_product_two()
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("strict_pass") else 1
    if args.command == "lock-product-two":
        result = app.lock_product_two(args.release_id)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "product-two":
        stages = ["context", "mapping", "derivation", "splits", "indexes", "training", "validation"]
        start_index = stages.index(args.from_stage)
        stop_index = stages.index(args.stop_after)
        if start_index > stop_index:
            parser.error("--from-stage must not come after --stop-after")
        selected = stages[start_index : stop_index + 1]
        report: dict[str, Any] = {"baseline": app.freeze_product_one(), "selected_stages": selected}
        if "context" in selected:
            report["context"] = app.build_contextual_canonical(
                prefer_parquet=not args.csv_fallback,
                resume=args.resume,
            )
        if "mapping" in selected:
            report["mapping"] = app.map_reactions(
                backend=args.mapping_backend,
                fallback_backend=None if args.fallback_backend == "none" else args.fallback_backend,
                allow_auto_fallback=args.allow_auto_fallback,
                batch_size=args.batch_size,
                shard_size=args.shard_size,
                resume=args.resume,
                max_rows=args.max_rows,
                prefer_parquet=not args.csv_fallback,
            )
        if "derivation" in selected:
            report["derivation"] = app.derive_reaction_centres(
                resume=args.resume,
                include_mcs=args.include_mcs,
                max_rows=args.max_rows,
                prefer_parquet=not args.csv_fallback,
            )
        if "splits" in selected:
            report["splits"] = app.rebuild_product_two_splits(
                prefer_parquet=not args.csv_fallback,
                random_seed=42,
            )
        if "indexes" in selected:
            report["indexes"] = app.build_contextual_index(args.max_rows)
            report["anomaly_model"] = app.build_anomaly_model()
        if "training" in selected:
            report["classification_models"] = app.trainer(
                "uspto_multistep_contextual_v2", args.max_rows
            ).train_many(args.classification_tasks, promote_validated=args.request_promotion)
            specialist_trainer = app.specialist_trainer(args.max_rows)
            report["specialist_models"] = {
                task: specialist_trainer.train(task, request_promotion=args.request_promotion)
                for task in args.specialist_tasks
            }
        if "validation" in selected:
            report["acceptance"] = app.validate_product_two()
        if args.lock_release:
            if "validation" not in selected:
                parser.error("--lock-release requires validation in the selected stage range")
            if not report.get("acceptance", {}).get("strict_pass"):
                report["release_lock"] = {"locked": False, "reason": "Strict scientific acceptance did not pass."}
            else:
                report["release_lock"] = app.lock_product_two(args.release_id)
        report_path = cfg.reports_dir / "product_two_run.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"report": report_path.relative_to(cfg.project_root).as_posix(), **report}, indent=2, default=str))
        return 0 if report.get("acceptance", {}).get("strict_pass", True) else 1
    if args.command == "export-hf":
        canonical_dir = cfg.canonical_v2_dir if args.contextual else cfg.canonical_dir
        result = build_huggingface_bundle(
            canonical_dir=canonical_dir,
            destination=Path(args.destination),
            reports_dir=cfg.reports_dir,
            include_models_from=cfg.model_dir if args.include_models else None,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.command == "serve":
        uvicorn.run("reacts.api.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
