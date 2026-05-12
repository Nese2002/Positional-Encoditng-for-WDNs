#python3 plot_distributions.py \
  #--train_path datasets/ctown.zip \
  #--test_synthetic_path datasets/ctown.zip \
  #--test_timeseries_path datasets/ctown_test_24h.zip \
  #--network_name C-Town \
  #--attribute pressure \
  #--num_samples 2000

"""
Plot density distributions of training and test sets
to verify compliance with paper's Figure 9.

Usage:
    python3 plot_distributions.py \
        --train_path datasets/ctown.zip \
        --test_synthetic_path datasets/ctown.zip \
        --test_timeseries_path datasets/ctown_test_24h.zip \
        --network_name C-Town \
        --num_samples 2000
"""

import numpy as np
import zarr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse


def load_samples(zip_path: str, split: str, attribute: str, num_samples: int) -> np.ndarray:
    """Load random samples from a zarr dataset split."""
    store = zarr.ZipStore(zip_path, mode='r')
    root = zarr.open_group(store, mode='r')
    
    data = root[attribute][split][:]  # shape: [num_snapshots, num_nodes]
    
    # Sample random snapshots
    total = data.shape[0]
    indices = np.random.choice(total, min(num_samples, total), replace=False)
    sampled = data[indices]  # shape: [num_samples, num_nodes]
    
    store.close()
    return sampled.flatten()  # flatten all nodes and snapshots


def plot_kde(ax, values, label, color, linestyle='-', alpha=0.3):
    """Plot a KDE density curve."""
    kde = gaussian_kde(values, bw_method=0.1)
    x_min, x_max = values.min(), values.max()
    x = np.linspace(x_min, x_max, 500)
    y = kde(x)
    ax.plot(x, y, color=color, linestyle=linestyle, label=label, linewidth=1.5)
    ax.fill_between(x, y, alpha=alpha, color=color)


def plot_distributions(
    train_path: str,
    test_synthetic_path: str,
    test_timeseries_path: str,
    network_name: str = "C-Town",
    attribute: str = "pressure",
    num_samples: int = 2000,
    output_path: str = "distribution_comparison.png",
):
    print(f"Loading training data from {train_path}...")
    train_samples = load_samples(train_path, 'train', attribute, num_samples)

    print(f"Loading synthetic test data from {test_synthetic_path}...")
    test_synthetic_samples = load_samples(test_synthetic_path, 'test', attribute, num_samples)

    print(f"Loading time-series test data from {test_timeseries_path}...")
    store = zarr.ZipStore(test_timeseries_path, mode='r')
    root = zarr.open_group(store, mode='r')
    test_ts_data = root[attribute]['test'][:]
    store.close()
    total = test_ts_data.shape[0]
    indices = np.random.choice(total, min(num_samples, total), replace=False)
    test_ts_samples = test_ts_data[indices].flatten()

    print(f"\nDataset summary:")
    print(f"  Training samples (flattened):          {len(train_samples)}")
    print(f"  Synthetic test samples (flattened):    {len(test_synthetic_samples)}")
    print(f"  Time-series test samples (flattened):  {len(test_ts_samples)}")
    print(f"\n  Training   - mean: {train_samples.mean():.2f}, std: {train_samples.std():.2f}")
    print(f"  Synth test - mean: {test_synthetic_samples.mean():.2f}, std: {test_synthetic_samples.std():.2f}")
    print(f"  TS test    - mean: {test_ts_samples.mean():.2f}, std: {test_ts_samples.std():.2f}")

    # Check if distributions are identical (paper's key finding)
    from scipy.stats import ks_2samp
    stat_synth, p_synth = ks_2samp(train_samples, test_synthetic_samples)
    stat_ts, p_ts = ks_2samp(train_samples, test_ts_samples)
    print(f"\nKolmogorov-Smirnov test (train vs synthetic test): stat={stat_synth:.4f}, p={p_synth:.4f}")
    print(f"Kolmogorov-Smirnov test (train vs time-series test): stat={stat_ts:.4f}, p={p_ts:.4f}")
    print(f"\n→ Train vs Synthetic: {'IDENTICAL ✅ (as expected)' if p_synth > 0.05 else 'DIFFERENT ⚠️'}")
    print(f"→ Train vs Time-series: {'DIFFERENT ✅ (as expected)' if p_ts < 0.05 else 'SIMILAR ⚠️ (unexpected)'}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    plot_kde(ax, train_samples,
             label='Simulation-Training',
             color='steelblue', linestyle='-', alpha=0.3)

    plot_kde(ax, test_synthetic_samples,
             label='Simulation-Test',
             color='darkorange', linestyle='--', alpha=0.1)

    plot_kde(ax, test_ts_samples,
             label='Time demand patterns-Test',
             color='green', linestyle='-', alpha=0.3)

    ax.set_xlabel('Pressure head (m)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'({network_name}) Density distribution of training and test sets', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="datasets/ctown.zip")
    parser.add_argument("--test_synthetic_path", default="datasets/ctown.zip")
    parser.add_argument("--test_timeseries_path", default="datasets/ctown_test_24h.zip")
    parser.add_argument("--network_name", default="C-Town")
    parser.add_argument("--attribute", default="pressure",
                        choices=["pressure", "head"])
    parser.add_argument("--num_samples", default=2000, type=int)
    parser.add_argument("--output_path", default="distribution_comparison.png")
    args = parser.parse_args()

    plot_distributions(
        train_path=args.train_path,
        test_synthetic_path=args.test_synthetic_path,
        test_timeseries_path=args.test_timeseries_path,
        network_name=args.network_name,
        attribute=args.attribute,
        num_samples=args.num_samples,
        output_path=args.output_path,
    )