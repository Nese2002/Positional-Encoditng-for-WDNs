"""
Explore the ctown.zip dataset structure and statistics.
Run: python3 explore_dataset.py --dataset_path datasets/ctown.zip
"""

import zarr
import numpy as np
import argparse
import json

def explore_dataset(dataset_path: str):
    print("=" * 60)
    print(f"DATASET: {dataset_path}")
    print("=" * 60)

    store = zarr.ZipStore(dataset_path, mode='r')
    root = zarr.open_group(store, mode='r')

    # ─── 1. TREE STRUCTURE ───────────────────────────────────────
    print("\n── 1. STRUCTURE ──────────────────────────────────────────")
    print(root.tree())

    # ─── 2. METADATA ─────────────────────────────────────────────
    print("\n── 2. METADATA ───────────────────────────────────────────")
    if root.attrs:
        attrs = dict(root.attrs)
        # Print args
        if 'args' in attrs:
            print("\n  Generation arguments:")
            for k, v in attrs['args'].items():
                if v is not None and v is not False:
                    print(f"    {k}: {v}")
        # Print config sections
        if 'config' in attrs:
            print("\n  Config sections:")
            for section, values in attrs['config'].items():
                print(f"    [{section}]")
                for k, v in values.items():
                    print(f"      {k} = {v}")
        # Print node ordering
        if 'ordered_names_by_attr' in attrs:
            names = attrs['ordered_names_by_attr']
            for attr, node_list in names.items():
                print(f"\n  Node order for '{attr}': "
                      f"{node_list[:5]} ... {node_list[-5:]} "
                      f"(total: {len(node_list)})")

    # ─── 3. ARRAY SHAPES ─────────────────────────────────────────
    print("\n── 3. ARRAY SHAPES ───────────────────────────────────────")
    for attr in root.keys():
        group = root[attr]
        if hasattr(group, 'keys'):
            print(f"\n  {attr}/")
            for split in group.keys():
                arr = group[split]
                print(f"    {split}: shape={arr.shape}, "
                      f"dtype={arr.dtype}, "
                      f"chunks={arr.chunks}")

    # ─── 4. STATISTICS ───────────────────────────────────────────
    print("\n── 4. STATISTICS (from training set) ─────────────────────")
    for attr in root.keys():
        group = root[attr]
        if hasattr(group, 'attrs') and group.attrs:
            a = dict(group.attrs)
            print(f"\n  {attr}:")
            for k in ['mean', 'std', 'min', 'max', 'cv', 'mcoef', 'bcoef']:
                if k in a:
                    print(f"    {k:8s} = {a[k]:.6f}")

    # ─── 5. DATA CONTENT ─────────────────────────────────────────
    print("\n── 5. DATA CONTENT ───────────────────────────────────────")
    for attr in root.keys():
        group = root[attr]
        if hasattr(group, 'keys'):
            print(f"\n  {attr}:")
            for split in ['train', 'valid', 'test']:
                if split in group:
                    arr = group[split][:]
                    print(f"    {split}:")
                    print(f"      rows (scenarios): {arr.shape[0]}")
                    print(f"      cols (nodes):     {arr.shape[1]}")
                    print(f"      mean:  {arr.mean():.4f} m")
                    print(f"      std:   {arr.std():.4f} m")
                    print(f"      min:   {arr.min():.4f} m")
                    print(f"      max:   {arr.max():.4f} m")
                    print(f"      % negative: "
                          f"{(arr < 0).mean() * 100:.2f}%")

    # ─── 6. PER-NODE STATISTICS ──────────────────────────────────
    print("\n── 6. PER-NODE STATISTICS (training set) ─────────────────")
    for attr in root.keys():
        group = root[attr]
        if 'train' in group:
            arr = group['train'][:]  # shape: [scenarios, nodes]
            node_means = arr.mean(axis=0)   # mean per node
            node_stds  = arr.std(axis=0)    # std per node
            node_mins  = arr.min(axis=0)
            node_maxs  = arr.max(axis=0)

            print(f"\n  {attr} (per node across all scenarios):")
            print(f"    Most pressurized node:  "
                  f"{node_means.max():.2f} m (mean)")
            print(f"    Least pressurized node: "
                  f"{node_means.min():.2f} m (mean)")
            print(f"    Most variable node:     "
                  f"{node_stds.max():.2f} m (std)")
            print(f"    Least variable node:    "
                  f"{node_stds.min():.2f} m (std)")

            # Top 5 highest pressure nodes
            top5_idx = np.argsort(node_means)[-5:][::-1]
            bot5_idx = np.argsort(node_means)[:5]
            print(f"    Top 5 node indices by mean pressure: {top5_idx.tolist()}")
            print(f"    Bot 5 node indices by mean pressure: {bot5_idx.tolist()}")

    # ─── 7. SPLIT RATIO CHECK ────────────────────────────────────
    print("\n── 7. SPLIT RATIO CHECK ──────────────────────────────────")
    for attr in root.keys():
        group = root[attr]
        if hasattr(group, 'keys'):
            sizes = {}
            for split in ['train', 'valid', 'test']:
                if split in group:
                    sizes[split] = group[split].shape[0]
            if sizes:
                total = sum(sizes.values())
                print(f"\n  {attr}:")
                for split, n in sizes.items():
                    print(f"    {split}: {n} scenarios "
                          f"({n/total*100:.1f}%)")
                print(f"    total: {total} scenarios")

    store.close()
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        default="datasets/ctown.zip",
                        type=str)
    args = parser.parse_args()
    explore_dataset(args.dataset_path)