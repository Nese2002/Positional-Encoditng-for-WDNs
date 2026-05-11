#To run: python3 generate_test_set.py \
    #--inp_path inputs/ctown.inp \
    #--output_path datasets/ctown_test_24h.zip \
    #--timestep_minutes 5

import numpy as np
import zarr
import epynet
from epynet import Network
import wntr

def generate_24h_test_dataset(
    inp_path: str,
    output_path: str,
    timestep_minutes: int = 5,
    add_noise: bool = False,
    noise_mean: float = 0.1,
    noise_std: float = 1.0,
):
    """
    Generate a 24-hour time-series test dataset from a .inp file
    with real demand patterns, matching the paper's evaluation protocol.
    
    Args:
        inp_path: Path to the .inp network file
        output_path: Path to save the output .zip zarr store
        timestep_minutes: Sampling interval in minutes (default 5)
        add_noise: Whether to inject Gaussian noise (noisy test)
        noise_mean: Mean of Gaussian noise on demands
        noise_std: Std of Gaussian noise on demands
    """
    
    # Load network with WNTR for time-series simulation
    print(f"Loading network from {inp_path}...")
    wn = wntr.network.WaterNetworkModel(inp_path)
    
    # Set simulation duration to 24 hours
    duration_hours = 24
    duration_seconds = duration_hours * 3600
    timestep_seconds = timestep_minutes * 60
    
    wn.options.time.duration = duration_seconds
    wn.options.time.hydraulic_timestep = timestep_seconds
    wn.options.time.report_timestep = timestep_seconds
    
    print(f"Simulation duration: {duration_hours} hours")
    print(f"Timestep: {timestep_minutes} minutes")
    
    num_snapshots = duration_seconds // timestep_seconds
    print(f"Expected snapshots: {num_snapshots}")
    
    # Inject Gaussian noise into demands if requested
    if add_noise:
        print(f"Injecting Gaussian noise: mean={noise_mean}, std={noise_std}")
        for junction_name, junction in wn.junctions():
            base_demand = junction.base_demand
            noise = np.random.normal(
                loc=noise_mean * base_demand,
                scale=noise_std * abs(base_demand)
            )
            junction.base_demand = max(0, base_demand + noise)
    
    # Run hydraulic simulation
    print("Running EPANET hydraulic simulation...")
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    # Extract pressure and head at each timestep
    pressure_results = results.node['pressure'].values  # shape: [num_snapshots, num_nodes]
    head_results = results.node['head'].values          # shape: [num_snapshots, num_nodes]
    node_names = list(results.node['pressure'].columns)
    
    # Filter out reservoir nodes (keep junctions only)
    reservoir_names = [name for name, _ in wn.reservoirs()]
    tank_names = [name for name, _ in wn.tanks()]
    junction_names = [name for name, _ in wn.junctions()]
    
    # Keep only junction nodes
    junction_mask = [name in junction_names for name in node_names]
    junction_indices = [i for i, keep in enumerate(junction_mask) if keep]
    
    pressure_junctions = pressure_results[:, junction_indices]
    head_junctions = head_results[:, junction_indices]
    
    print(f"Pressure shape: {pressure_junctions.shape}")
    print(f"Head shape: {head_junctions.shape}")
    print(f"Num junctions: {len(junction_names)}")
    
    # Remove invalid snapshots (NaN or negative pressures)
    valid_mask = ~(
        np.isnan(pressure_junctions).any(axis=1) |
        (pressure_junctions.min(axis=1) < -1e-3)
    )
    pressure_junctions = pressure_junctions[valid_mask]
    head_junctions = head_junctions[valid_mask]
    num_valid = valid_mask.sum()
    print(f"Valid snapshots: {num_valid}/{num_snapshots}")
    
    # Save to Zarr format (compatible with the training dataset format)
    print(f"Saving to {output_path}...")
    store = zarr.ZipStore(output_path, mode='w')
    root = zarr.open_group(store, mode='w')
    
    # Save as a single 'test' split
    # pressure
    pressure_group = root.create_group('pressure')
    pressure_group.create_dataset(
        'test',
        data=pressure_junctions.astype(np.float64),
        chunks=(8, pressure_junctions.shape[1])
    )
    
    # head
    head_group = root.create_group('head')
    head_group.create_dataset(
        'test',
        data=head_junctions.astype(np.float64),
        chunks=(8, head_junctions.shape[1])
    )
    
    # Save metadata
    root.attrs['node_names'] = junction_names
    root.attrs['num_snapshots'] = int(num_valid)
    root.attrs['timestep_minutes'] = timestep_minutes
    root.attrs['noisy'] = add_noise
    root.attrs['source_inp'] = inp_path
    
    store.close()
    print(f"Saved {num_valid} snapshots to {output_path}")
    print(f"Pressure range: [{pressure_junctions.min():.2f}, {pressure_junctions.max():.2f}] m")
    return num_valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_path", default="inputs/ctown.inp")
    parser.add_argument("--output_path", default="datasets/ctown_test_24h.zip")
    parser.add_argument("--timestep_minutes", default=5, type=int)
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--noise_mean", default=0.1, type=float)
    parser.add_argument("--noise_std", default=1.0, type=float)
    args = parser.parse_args()
    
    # Generate clean test set
    generate_24h_test_dataset(
        inp_path=args.inp_path,
        output_path=args.output_path,
        timestep_minutes=args.timestep_minutes,
        add_noise=args.add_noise,
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
    )