"""Train the QuantGAN TCN baseline on the prepared NIFTY-50 market series."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASELINES_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = BASELINES_DIR / "config_quantgan_smoke.json"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_windows(path: Path, window: int, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(np.load(path), dtype=np.float32)
    if windows.ndim != 2 or windows.shape[1] != window:
        raise ValueError(f"Expected (samples, {window}) windows, got {windows.shape}")
    if not np.isfinite(windows).all():
        raise ValueError("Input contains NaN or Inf values")
    split = int(len(windows) * train_fraction)
    if split < 2 or split >= len(windows):
        raise ValueError("train_fraction must leave non-empty train and test sets")
    return windows[:split], windows[split:]


def gaussianize_windows(
    train: np.ndarray, test: np.ndarray, enabled: bool
) -> tuple[np.ndarray, np.ndarray, object | None]:
    if not enabled:
        return train, test, None
    repo = BASELINES_DIR / "external" / "QuantGANs-replication"
    sys.path.insert(0, str(repo))
    from backend.gaussianize import Gaussianize

    transformer = Gaussianize(strategy="lambert")
    transformer.fit(train.reshape(-1, 1))
    train_transformed = transformer.transform(train.reshape(-1, 1)).reshape(train.shape)
    test_transformed = transformer.transform(test.reshape(-1, 1)).reshape(test.shape)
    return train_transformed.astype(np.float32), test_transformed.astype(np.float32), transformer


def build_models(config: dict):
    repo = BASELINES_DIR / "external" / "QuantGANs-replication"
    sys.path.insert(0, str(repo))
    try:
        from backend.gan import GAN
        from backend.tcn import make_TCN
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "QuantGAN requires TensorFlow and TensorFlow Addons. "
            "Use the documented Python 3.10 environment and install the QuantGAN dependencies."
        ) from error

    input_dim = (1, config["window"], 1)
    generator_context = config["window"] + config["block_size"] * sum(config["dilations"])
    noise_dim = (1, generator_context, config["noise_features"])
    generator = make_TCN(
        config["dilations"], config["fixed_filters"], config["moving_filters"],
        use_batchNorm=False, one_series_output=True, sigmoid=False,
        input_dim=noise_dim, block_size=config["block_size"],
    )
    discriminator = make_TCN(
        config["dilations"], config["fixed_filters"], config["moving_filters"],
        use_batchNorm=False, one_series_output=True, sigmoid=False,
        input_dim=input_dim, block_size=config["block_size"],
    )
    gan = GAN(
        discriminator=discriminator,
        generator=generator,
        training_input=config["window"],
        lr_d=config["learning_rate_discriminator"],
        lr_g=config["learning_rate_generator"],
    )
    # The replication assumes a 3D generator input; this TCN uses
    # (series, time, features), so override its inferred noise shape.
    gan.noise_shape = [1, generator_context, config["noise_features"]]
    return gan


def generate(gan, count: int, config: dict) -> np.ndarray:
    import tensorflow as tf

    noise_shape = [count, *gan.noise_shape]
    generated = gan.generator(tf.random.normal(noise_shape), training=False).numpy()
    generated = np.asarray(generated, dtype=np.float32)
    if generated.ndim == 4:
        generated = generated[:, 0, :, 0]
    if generated.shape != (count, config["window"]):
        raise ValueError(f"Unexpected generated shape {generated.shape}")
    if not np.isfinite(generated).all():
        raise ValueError("Generated data contains NaN or Inf values")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = Path(config["input_file"])
    if not input_path.is_absolute():
        input_path = BASELINES_DIR.parent / input_path
    output_dir = Path(config["output_directory"])
    if not output_dir.is_absolute():
        output_dir = BASELINES_DIR.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(config["seed"])
    train, test = load_windows(input_path, config["window"], config["train_fraction"])
    train, test, transformer = gaussianize_windows(train, test, config["gaussianize"])
    train_data = train[:, None, :, None]
    gan = build_models(config)
    start = datetime.now(timezone.utc)
    gan.train(
        train_data,
        batch_size=config["batch_size"],
        n_batches=config["steps"],
        additional_d_steps=config["additional_discriminator_steps"],
    )
    generated = generate(gan, len(test), config)
    if transformer is not None:
        generated = np.clip(generated, -config["inverse_clip"], config["inverse_clip"])
        mu, sigma, delta = transformer.coefs_[0]
        normalized = (generated - mu) / sigma
        exponent = np.clip(normalized * normalized * (delta * 0.5), None, 80.0)
        with np.errstate(over="ignore", invalid="ignore"):
            generated = (mu + sigma * normalized * np.exp(exponent)).astype(np.float32)
    if not np.isfinite(generated).all():
        raise ValueError("Inverse-transformed generated data contains NaN or Inf values")
    if np.max(np.abs(generated)) > 1.0:
        raise ValueError("Generated returns exceed the sanity bound of 1.0")
    gan.generator.save_weights(str(output_dir / "generator.weights.h5"))
    gan.discriminator.save_weights(str(output_dir / "discriminator.weights.h5"))
    np.save(output_dir / "generated_windows.npy", generated)
    end = datetime.now(timezone.utc)
    metadata = {
        "status": "completed",
        "start_time_utc": start.isoformat(),
        "end_time_utc": end.isoformat(),
        "duration_seconds": (end - start).total_seconds(),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "generated_shape": list(generated.shape),
        "config": config,
        "generated_mean": float(generated.mean()),
        "generated_std": float(generated.std()),
        "generated_min": float(generated.min()),
        "generated_max": float(generated.max()),
        "checkpoint": ["generator.weights.h5", "discriminator.weights.h5"],
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()