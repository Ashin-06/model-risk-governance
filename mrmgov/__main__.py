"""
CLI entry point for `mrmgov`:
  - `python -m mrmgov bench`
  - `python -m mrmgov serve`    (settings: MRMGOV_* environment variables, see ops.py)
  - `python -m mrmgov migrate`  (apply database migrations to MRMGOV_DATABASE_URL and exit)
  - `python -m mrmgov realdata --data FILE` (UCI credit default data -> results/real_data.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .bench import run_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrmgov",
        description="Model risk checks: drift, conformal abstention, group fairness",
    )
    sub = parser.add_subparsers(dest="command")

    bench_p = sub.add_parser("bench", help="Run the benchmark on generated data")
    bench_p.add_argument("--seed", type=int, default=42, help="Random seed (0 to 2**32 - 1)")
    bench_p.add_argument("--json", action="store_true", help="Output full JSON report")

    serve_p = sub.add_parser("serve", help="Start FastAPI governance server")
    serve_p.add_argument("--host", help="default MRMGOV_HOST or 127.0.0.1; any other address needs API keys")
    serve_p.add_argument("--port", type=int, help="default MRMGOV_PORT or 8019")
    sub.add_parser("migrate", help="apply database migrations to MRMGOV_DATABASE_URL")

    real_p = sub.add_parser("realdata", help="Run the checks on the UCI credit-card default data")
    real_p.add_argument("--data", required=True, help='"default of credit card clients.xls" (or a CSV export)')
    real_p.add_argument("--out", default="results/real_data.json")

    args = parser.parse_args(argv)
    if args.command == "realdata":
        from .realdata import main as realdata

        realdata(args.data, args.out)
        return 0
    if args.command == "migrate":
        from . import db, ops

        applied = db.migrate(db.engine(ops.load_settings().database_url))
        print(*(applied or ["up to date"]), sep="\n")
        return 0
    if args.command == "serve":
        import uvicorn

        from . import ops

        for name, val in (("HOST", args.host), ("PORT", args.port)):
            if val is not None:
                os.environ[f"MRMGOV_{name}"] = str(val)
        try:
            settings = ops.load_settings()  # fail here, before uvicorn, with a readable message
        except ops.ConfigError as e:
            sys.exit(f"configuration error: {e}")
        ops.setup_logging(settings.log_level)
        uvicorn.run("mrmgov.api:app", host=settings.host, port=settings.port, log_config=None, access_log=False)
        return 0

    # Default or `bench` command
    seed = getattr(args, "seed", 42)
    if not 0 <= seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    as_json = getattr(args, "json", False)
    report = run_benchmark_suite(seed=seed)

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        cov_d = report["drift_scenarios"]["covariate_drift"]
        conc_d = report["drift_scenarios"]["concept_drift"]
        conf = report["conformal"]["result"]
        fair = report["fairness"]
        cert = report["certificate"]

        print("=" * 78)
        print("  mrmgov benchmark (generated data, seed %d)" % seed)
        print("=" * 78)
        print(
            f"[1] Covariate Drift Scenario : regime={cov_d['regime']} | "
            f"identified={cov_d['covariate_drifted_features']}"
        )
        print(
            f"[2] Concept Drift Scenario   : regime={conc_d['regime']}   | "
            f"identified={conc_d['concept_drifted_features']}"
        )
        mean_cov = report["conformal"]["mean_group_coverage"]
        print(
            f"[3] Mondrian Conformal (95%) : mean group coverage over "
            f"{report['conformal']['n_splits']} splits "
            f"{min(mean_cov.values()):.4f}-{max(mean_cov.values()):.4f} | "
            f"abstain={conf['overall_abstention_rate']:.4f} (first split)"
        )
        print(
            f"[4] Group-threshold search   : DIR {fair['baseline']['disparate_impact_ratio']:.4f} -> "
            f"{fair['repaired']['disparate_impact_ratio']:.4f} | "
            f"EOD {fair['baseline']['equal_opportunity_diff']:+.4f} -> "
            f"{fair['repaired']['equal_opportunity_diff']:+.4f} | "
            f"AUC Retention={fair['auc_retention_ratio']*100:.2f}% | "
            f"Profit Retention={fair['profit_retention_ratio']*100:.2f}%"
        )
        print(
            f"[5] Governance Certificate   : id={cert['certificate_id']} | "
            f"disposition={cert['overall_disposition']}"
        )
        print(f"    Root SHA-256             : {cert['certificate_root_hash']}")
        print("-" * 78)
        for k, v in report["assertions"].items():
            mark = "PASS" if v else "FAIL"
            print(f"  [{mark}] {k}")
        print("=" * 78)

    return 0 if report["benchmark_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
