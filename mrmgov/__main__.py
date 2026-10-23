"""
CLI entry point for `mrmgov`:
  - `python -m mrmgov bench`
  - `python -m mrmgov serve --port 8019`
"""

from __future__ import annotations

import argparse
import json
import sys
import uvicorn

from .bench import run_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mrmgov",
        description="SR 11-7 / RBI Model Risk Governance Engine CLI",
    )
    sub = parser.add_subparsers(dest="command")

    bench_p = sub.add_parser("bench", help="Run SR 11-7 / RBI MRM benchmark suite")
    bench_p.add_argument("--seed", type=int, default=42, help="Random seed")
    bench_p.add_argument("--json", action="store_true", help="Output full JSON report")

    serve_p = sub.add_parser("serve", help="Start FastAPI governance server")
    serve_p.add_argument("--host", type=str, default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8019)

    args = parser.parse_args(argv)
    if args.command == "serve":
        uvicorn.run("mrmgov.api:app", host=args.host, port=args.port, reload=False)
        return 0

    # Default or `bench` command
    seed = getattr(args, "seed", 42)
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
        print("  SR 11-7 / RBI MODEL RISK GOVERNANCE ENGINE (mrmgov) — BENCHMARK REPORT")
        print("=" * 78)
        print(
            f"[1] Covariate Drift Scenario : regime={cov_d['regime']} | "
            f"identified={cov_d['covariate_drifted_features']}"
        )
        print(
            f"[2] Concept Drift Scenario   : regime={conc_d['regime']}   | "
            f"identified={conc_d['concept_drifted_features']}"
        )
        print(
            f"[3] Mondrian Conformal (95%) : overall_cov={conf['overall_coverage']:.4f} | "
            f"min_group_cov={report['conformal']['min_group_coverage']:.4f} | "
            f"abstain={conf['overall_abstention_rate']:.4f}"
        )
        print(
            f"[4] ECOA Fair-Lending Repair : DIR {fair['baseline']['disparate_impact_ratio']:.4f} -> "
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
