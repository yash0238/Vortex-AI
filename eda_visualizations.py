"""Backward-compatible entry point for the EDA dashboard."""

from eda_dashboard import plot_eda


def create_dashboard():
    """Generate the dashboard using the canonical implementation."""
    return plot_eda()


if __name__ == "__main__":
    create_dashboard()