"""MetroGuard command line interface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from metroguard.config import load_config
from metroguard.data import download_dataset, prepare_features
from metroguard.pipeline import read_metrics, train_all

app = typer.Typer(help="MetroGuard reproducible predictive-maintenance toolkit.")
data_app = typer.Typer(help="Download and prepare MetroPT-3.")
app.add_typer(data_app, name="data")

ConfigOption = Annotated[
    Path,
    typer.Option("--config", "-c", help="Path to the project YAML configuration."),
]


@data_app.command("download")
def download(
    config_path: ConfigOption = Path("configs/default.yaml"),
    force: Annotated[bool, typer.Option(help="Re-download the official archive.")] = False,
) -> None:
    metadata = download_dataset(load_config(config_path), force=force)
    typer.echo(f"Downloaded and verified MetroPT-3: {metadata['sha256']}")


@data_app.command("prepare")
def prepare(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    metadata = prepare_features(load_config(config_path))
    typer.echo(
        f"Prepared {metadata['feature_rows']:,} causal bins from {metadata['raw_rows']:,} rows."
    )


@app.command("train")
def train(
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Train the fixed four-model benchmark."),
    ] = False,
    config_path: ConfigOption = Path("configs/default.yaml"),
) -> None:
    if not all_models:
        typer.echo("The v1 protocol is fixed; pass --all to run every pre-registered model.")
        raise typer.Exit(code=2)
    metrics = train_all(load_config(config_path))
    primary = metrics["models"][metrics["primary_model"]]
    typer.echo(
        "Training complete — "
        f"early recall {primary['early_event_recall']}, "
        f"false alarms/day {primary['false_alarms_per_day']:.3f}."
    )


@app.command("evaluate")
def evaluate(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    metrics = read_metrics(load_config(config_path))
    primary = metrics["models"][metrics["primary_model"]]
    typer.echo(
        f"{metrics['primary_model']}: {primary['early_event_recall']} early events, "
        f"PR-AUC {primary['pr_auc_official_failure_intervals']:.4f}."
    )


@app.command("reproduce")
def reproduce(config_path: ConfigOption = Path("configs/default.yaml")) -> None:
    config = load_config(config_path)
    download_dataset(config)
    prepare_features(config)
    train_all(config)
    typer.echo("Full MetroGuard reproduction completed.")


@app.command("api")
def api(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "metroguard.api:app", "--host", host, "--port", str(port)],
        check=True,
    )


@app.command("dashboard")
def dashboard(port: Annotated[int, typer.Option()] = 8501) -> None:
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "src/metroguard/dashboard.py", "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":  # pragma: no cover
    app()

