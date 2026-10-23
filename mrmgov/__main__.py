"""
CLI entry point for `mrmgov`:
  - `python -m mrmgov bench`
  - `python -m mrmgov serve --port 8019`  (refuses a non-loopback --host without MRMGOV_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uvicorn

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
    serve_p.add_argument("--host", type=str, default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8019)

    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("MRMGOV_API_KEY"):
            print("Refusing to listen on a non-loopback address without MRMGOV_API_KEY set.", file=sys.stderr)
            return 2
        uvicorn.run("mrmgov.api:app", host=args.host, port=args.port, reload=False)
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
