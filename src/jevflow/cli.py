"""Command-line entry point: ``jevflow {api,dashboard,up,benchmark,scenarios}``."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from jevflow.config import get_settings

DASHBOARD = Path(__file__).parent / "dashboard" / "app.py"
STREAMLIT_THEME = [
    "--theme.base=dark",
    "--theme.primaryColor=#3987e5",
    "--theme.backgroundColor=#0b0e13",
    "--theme.secondaryBackgroundColor=#151923",
    "--theme.textColor=#e8e9ee",
    "--browser.gatherUsageStats=false",
    "--server.headless=true",
]


def _api(args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "jevflow.api.app:app",
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        log_level="info",
    )


def _dashboard_cmd(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD),
        f"--server.port={port}",
        *STREAMLIT_THEME,
    ]


def _dashboard(args: argparse.Namespace) -> None:
    raise SystemExit(subprocess.call(_dashboard_cmd(args.port)))  # noqa: S603 - fixed argv


def _up(args: argparse.Namespace) -> None:
    settings = get_settings()
    api = subprocess.Popen(  # noqa: S603 - fixed argv
        [
            sys.executable,
            "-m",
            "uvicorn",
            "jevflow.api.app:app",
            "--host",
            settings.api_host,
            "--port",
            str(settings.api_port),
        ]
    )
    try:
        subprocess.call(_dashboard_cmd(args.port))  # noqa: S603 - fixed argv
    finally:
        api.terminate()
        api.wait(timeout=10)


def _benchmark(args: argparse.Namespace) -> None:
    from jevflow.persistence.repository import Repository
    from jevflow.runtime.manager import run_headless
    from jevflow.simulation.scenarios import get_scenario

    settings = get_settings()
    scenario = get_scenario(args.scenario)
    if args.duration:
        scenario = scenario.model_copy(update={"duration_s": args.duration})
    controllers = args.controllers or [
        "fixed_time",
        "actuated",
        "max_pressure",
        *(["jev"] if settings.jev_enabled else []),
    ]
    repo = Repository(settings.database_path) if args.save else None
    print(
        f"\n  {scenario.name}: {scenario.tagline}\n  seed {args.seed} · {scenario.duration_s / 60:.0f} min simulated\n"
    )
    header = f"  {'controller':<14}{'delay':>8}{'LOS':>5}{'p95':>8}{'stops':>7}{'veh/h':>7}{'max wait':>10}{'fair':>6}{'Jev ms':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for controller in controllers:
        _, summary, error = run_headless(scenario, controller, args.seed, settings, repo)
        if error or summary is None:
            print(f"  {controller:<14} failed: {error}")
            continue
        latency = f"{summary.jev_avg_latency_ms:.0f}" if summary.jev_avg_latency_ms is not None else "–"
        print(
            f"  {controller:<14}{summary.avg_control_delay_s:>7.1f}s{summary.level_of_service:>5}"
            f"{summary.p95_delay_s:>7.0f}s{summary.stops_per_vehicle:>7.2f}{summary.throughput_vph:>7.0f}"
            f"{summary.max_wait_s:>9.0f}s{summary.fairness_index:>6.2f}{latency:>8}"
        )
    print()


def _scenarios(_: argparse.Namespace) -> None:
    from jevflow.simulation.scenarios import SCENARIOS

    for s in SCENARIOS.values():
        print(f"  {s.key:<20} {s.name}: {s.tagline}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="jevflow", description="Real-time traffic-signal control with TypeSafe Jev"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("api", help="run the FastAPI backend")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.set_defaults(func=_api)

    p = sub.add_parser("dashboard", help="run the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=_dashboard)

    p = sub.add_parser("up", help="run the API and the dashboard together")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=_up)

    p = sub.add_parser("benchmark", help="compare controllers headlessly in the terminal")
    p.add_argument("--scenario", default="rush_hour")
    p.add_argument("--controllers", nargs="*")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration", type=float, help="override scenario duration (s)")
    p.add_argument("--save", action="store_true", help="store runs in the SQLite history")
    p.set_defaults(func=_benchmark)

    p = sub.add_parser("scenarios", help="list built-in scenarios")
    p.set_defaults(func=_scenarios)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
