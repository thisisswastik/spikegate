"""
SpikeGate — Main entry point.

Run `python main.py --help` for available commands, or use the package scripts:
  spikegate-eval      → python -m eval.run_eval
  spikegate-dashboard → streamlit run dashboard/app.py
"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="spikegate",
        description="SpikeGate — Fraud spike detection and bounded agentic response",
    )
    subparsers = parser.add_subparsers(dest="command")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Run evaluation harness")
    eval_parser.add_argument("--hours", type=float, default=12.0)
    eval_parser.add_argument("--merchants", type=int, default=50)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--no-agent", action="store_true")
    eval_parser.add_argument("--report-path", default="eval/report.md")

    # dashboard subcommand
    dash_parser = subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501)

    # seed subcommand
    seed_parser = subparsers.add_parser("seed", help="Seed merchant context database")
    seed_parser.add_argument("--merchants", type=int, default=50)
    seed_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "eval":
        from dotenv import load_dotenv
        load_dotenv()
        from eval.run_eval import run_eval
        run_eval(
            simulation_hours=args.hours,
            n_merchants=args.merchants,
            seed=args.seed,
            report_path=args.report_path,
            use_agent=not args.no_agent,
        )

    elif args.command == "dashboard":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "dashboard/app.py",
            "--server.port", str(args.port),
        ])

    elif args.command == "seed":
        from agent.seed_context_db import seed_from_generator
        seed_from_generator(
            db_path="data/merchant_context.db",
            n_merchants=args.merchants,
            seed=args.seed,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
