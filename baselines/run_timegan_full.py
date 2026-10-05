#!/usr/bin/env python3
"""Run the production TimeGAN baseline on all NIFTY-50 windows."""

import argparse
import io
import json
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASELINES_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASELINES_DIR / "data" / "timegan_windows.npy"
DEFAULT_OUTPUT = BASELINES_DIR / "results" / "timegan_full"
EXPECTED_SHAPE = (3640, 60, 50)
LOSS_PATTERN = re.compile(r"(?:e_loss|s_loss|d_loss|g_loss_u|g_loss_s|g_loss_v|e_loss_t0):\s*([^,\s]+)")


class Tee:
    """Write training output to both the terminal and the run log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def load_windows(input_path):
    data = np.load(input_path)
    if data.shape != EXPECTED_SHAPE:
        raise ValueError("Expected input shape {}, got {}".format(EXPECTED_SHAPE, data.shape))
    if data.dtype != np.float32:
        raise ValueError("Expected float32 input, got {}".format(data.dtype))
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN or Inf values")
    return [data[index] for index in range(data.shape[0])]


def write_json(path, value):
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def build_config(input_path, output_dir):
    return {
        "status": "prepared",
        "input_file": str(input_path),
        "input_shape": list(EXPECTED_SHAPE),
        "dataset_windows_used": EXPECTED_SHAPE[0],
        "sequence_length": EXPECTED_SHAPE[1],
        "feature_dim": EXPECTED_SHAPE[2],
        "timegan_parameters": {
            "module": "gru",
            "hidden_dim": 24,
            "num_layer": 3,
            "batch_size": 32,
            "iterations": 2000
        },
        "output_directory": str(output_dir),
        "artifacts": {
            "generated_samples": "generated_samples.npy",
            "training_config": "training_config.json",
            "training_log": "training.log",
            "statistics": "statistics.json",
            "checkpoint": None
        },
        "checkpoint_note": "The original TimeGAN function does not expose checkpoint saving; no checkpoint is produced.",
        "external_repository_modified": False
    }


def main():
    parser = argparse.ArgumentParser(description="Train the full NIFTY-50 TimeGAN baseline")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(input_path, output_dir)
    write_json(output_dir / "training_config.json", config)

    start = datetime.now(timezone.utc)
    log_path = output_dir / "training.log"
    with log_path.open("w") as log_handle:
        captured_output = io.StringIO()
        output = Tee(sys.stdout, log_handle, captured_output)
        try:
            with redirect_stdout(output), redirect_stderr(output):
                print("TimeGAN full baseline starting")
                print("Start time (UTC):", start.isoformat())
                data = load_windows(input_path)
                print("Validated input shape:", EXPECTED_SHAPE)
                print("Using all windows:", len(data))

                timegan_repo = BASELINES_DIR / "external" / "TimeGAN"
                sys.path.insert(0, str(timegan_repo))
                from timegan import timegan

                generated = timegan(data, config["timegan_parameters"])
                generated_array = np.asarray(generated, dtype=np.float32)
                expected_generated_shape = (EXPECTED_SHAPE[0], EXPECTED_SHAPE[1], EXPECTED_SHAPE[2])
                if generated_array.shape != expected_generated_shape:
                    raise ValueError("Expected generated shape {}, got {}".format(expected_generated_shape, generated_array.shape))
                if generated_array.size == 0 or not np.isfinite(generated_array).all():
                    raise ValueError("Generated data is empty or contains NaN/Inf values")

                loss_values = [float(value) for value in LOSS_PATTERN.findall(captured_output.getvalue())]
                if not loss_values:
                    raise ValueError("No training loss values were captured in the TimeGAN log")
                if not np.isfinite(loss_values).all():
                    raise ValueError("Training log contains a non-finite loss")

                np.save(output_dir / "generated_samples.npy", generated_array)
                end = datetime.now(timezone.utc)
                statistics = {
                    "status": "completed",
                    "start_time_utc": start.isoformat(),
                    "end_time_utc": end.isoformat(),
                    "training_duration_seconds": (end - start).total_seconds(),
                    "iterations": 2000,
                    "batch_size": 32,
                    "model_parameters": config["timegan_parameters"],
                    "generated_sample_count": int(generated_array.shape[0]),
                    "generated_shape": list(generated_array.shape),
                    "generated_mean": float(generated_array.mean()),
                    "generated_std": float(generated_array.std()),
                    "generated_min": float(generated_array.min()),
                    "generated_max": float(generated_array.max()),
                    "loss_values_checked": len(loss_values),
                    "losses_finite": True,
                    "checkpoint": None
                }
                write_json(output_dir / "statistics.json", statistics)
                print("Generated data validated and saved.")
                print("End time (UTC):", end.isoformat())
        except Exception as error:
            end = datetime.now(timezone.utc)
            failure = {
                "status": "failed",
                "start_time_utc": start.isoformat(),
                "end_time_utc": end.isoformat(),
                "training_duration_seconds": (end - start).total_seconds(),
                "error_type": type(error).__name__,
                "error": str(error),
                "step_d_loss_handling": "External TimeGAN was not modified; failure was captured in training.log.",
                "checkpoint": None
            }
            write_json(output_dir / "statistics.json", failure)
            traceback.print_exc(file=output)
            raise


if __name__ == "__main__":
    main()