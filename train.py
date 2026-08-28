"""Specification-compatible training entry point."""

from training.trainer import parse_args, train


if __name__ == "__main__":
    train(parse_args())