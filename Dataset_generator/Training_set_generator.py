#
# Created on Wed Oct 18 2023
# Copyright (c) 2023 Huy Truong
# ------------------------------
# Purpose: This file supports to generate dataset based on a config file
# Version: 7.1
# Note: You may need to increase/ decrease values in the config to get stable states
# Tip: Start with gen_demand=True, set off other gen_* flags
# Tip: set debug=True for more details
# Tip: don't change static hydraulic values
# ------------------------------
#

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_REPO_DIR = PROJECT_ROOT / "gnn-pressure-estimation"
GNN_SOURCE_DIR = GNN_REPO_DIR / "gnn_pressure_estimation"

if str(GNN_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_SOURCE_DIR))


import shutil
import zarr
from time import time
from tqdm import tqdm
from generator.EPYNET.TokenGeneratorByRange import *
from generator.EPYNET.Executorv7 import *
import ray
import pandas as pd
from ray.exceptions import RayError


@dataclass(frozen=True)
class GenerationConfig:
    config: str = r"configs/v7.1/ctown_7v1__EPYNET_config.ini"
    num_scenarios: int = 10000
    storage_dir: Optional[str] = None

    init_valve_state: Optional[int] = 1
    init_pipe_state: Optional[int] = None

    remove_pattern: bool = True
    remove_control: bool = False
    remove_rule: bool = False

    gen_demand: bool = True
    gen_res_total_head: bool = True
    gen_pump_speed: bool = True
    gen_pump_init_status: bool = True
    gen_tank_level: bool = True
    gen_valve_setting: bool = True
    gen_valve_init_status: bool = True
    gen_roughness: bool = True

    gen_elevation: bool = False
    gen_diameter: bool = False
    gen_length: bool = False
    gen_minorloss: bool = False
    gen_tank_elevation: bool = False
    gen_tank_diameter: bool = False
    gen_valve_diameter: bool = False
    gen_pump_length: bool = False

    replace_nonzero_basedmd: bool = False
    update_demand_json: Optional[str] = None
    update_elevation_json: Optional[str] = None
    update_pipe_roughness_json: Optional[str] = None
    update_pipe_diameter_json: Optional[str] = None
    update_pipe_length_json: Optional[str] = None
    update_pipe_minorloss_json: Optional[str] = None
    update_valve_init_status_json: Optional[str] = None
    update_valve_setting_json: Optional[str] = None
    update_valve_diameter_json: Optional[str] = None
    update_pump_init_status_json: Optional[str] = None
    update_pump_speed_json: Optional[str] = None
    update_pump_length_json: Optional[str] = None
    update_tank_level_json: Optional[str] = None
    update_tank_elevation_json: Optional[str] = None
    update_tank_diameter_json: Optional[str] = None
    update_res_total_head_json: Optional[str] = None

    ele_kmean_init: str = "k-means++"
    update_elevation_method: str = "ran_cluster"
    ele_std: float = 1.0
    dia_kmean_init: str = "k-means++"
    update_totalhead_method: Optional[str] = None

    debug: bool = False
    allow_error: bool = False
    convert_results_by_flow_unit: Optional[str] = "LPS"
    change_dmd_by_junc_indices_path: Optional[str] = None

    accept_warning_code: bool = False
    pressure_lowerbound: Optional[float] = 0.0
    pressure_upperbound: Optional[float] = 151.0
    flowrate_threshold: Optional[float] = None
    mean_cv_threshold: Optional[float] = None
    neighbor_std_threshold: Optional[float] = None

    batch_size: int = 8
    executors: int = 4
    att: str = "pressure,head"
    train_ratio: float = 0.6
    valid_ratio: float = 0.2
    skip_resevoir_result: bool = False


CONFIG = GenerationConfig()


def generate_dataset(run_config: GenerationConfig = CONFIG):
    os.chdir(GNN_REPO_DIR)
    args = run_config
    program_start = time()

    config = ConfigParser()
    config.read(args.config)
    config.set("general", "num_scenarios", str(args.num_scenarios))
    if args.storage_dir is not None:
        config.set("general", "storage_dir", args.storage_dir)

    wn_inp_path = config.get("general", "wn_inp_path")
    storage_dir = config.get("general", "storage_dir")

    zarr_storage_dir = os.path.join(storage_dir, "zarrays")
    os.makedirs(storage_dir, exist_ok=True)
    shutil.rmtree(path=storage_dir, ignore_errors=True)
    os.makedirs(zarr_storage_dir, exist_ok=False)

    saved_path = storage_dir
    num_scenarios = config.getint("general", "num_scenarios")
    backup_num_scenarios = num_scenarios * 10
    batch_size = args.batch_size
    num_executors = args.executors
    expected_attributes = args.att.strip().split(",")
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    num_batches = backup_num_scenarios // batch_size
    num_chunks = backup_num_scenarios // batch_size
    support_node_attr_keys = ["head", "pressure", "demand"]
    support_link_attr_keys = ["flow", "velocity"]
    support_keys = list(set(support_node_attr_keys).union(support_link_attr_keys))
    for a in expected_attributes:
        if a not in support_keys:
            raise AttributeError(f"{a} is not found or not supported!")

    wn = Network(wn_inp_path)
    skip_nodes = config.get("general", "skip_nodes").strip().split(",") if config.has_option("general", "skip_nodes") else None

    valve_type_dict = {}

    featlen_dict = dict()

    if len(wn.junctions) > 0:
        if args.gen_demand:
            featlen_dict[ParamEnum.JUNC_DEMAND] = len(wn.junctions)

        if args.gen_elevation:
            featlen_dict[ParamEnum.JUNC_ELEVATION] = len(wn.junctions)

    if len(wn.pipes) > 0:
        num_pipes = len(wn.pipes)
        if args.gen_roughness:
            featlen_dict[ParamEnum.PIPE_ROUGHNESS] = num_pipes
        if args.gen_diameter:
            featlen_dict[ParamEnum.PIPE_DIAMETER] = num_pipes
        if args.gen_length:
            featlen_dict[ParamEnum.PIPE_LENGTH] = num_pipes
        if args.gen_minorloss:
            featlen_dict[ParamEnum.PIPE_MINORLOSS] = num_pipes

    if len(wn.pumps) > 0:
        num_pumps = len(wn.pumps)
        if args.gen_pump_init_status:
            featlen_dict[ParamEnum.PUMP_STATUS] = num_pumps
        if args.gen_pump_speed:
            featlen_dict[ParamEnum.PUMP_SPEED] = num_pumps
        if args.gen_pump_length:
            featlen_dict[ParamEnum.PUMP_LENGTH] = num_pumps

    if len(wn.tanks) > 0:
        num_tanks = len(wn.tanks)
        if args.gen_tank_level:
            featlen_dict[ParamEnum.TANK_LEVEL] = num_tanks
        if args.gen_tank_elevation:
            featlen_dict[ParamEnum.TANK_ELEVATION] = num_tanks
        if args.gen_tank_diameter:
            featlen_dict[ParamEnum.TANK_DIAMETER] = num_tanks

    if len(wn.valves) > 0:
        num_valves = len(wn.valves)
        if args.gen_valve_init_status:
            featlen_dict[ParamEnum.VALVE_STATUS] = num_valves
        if args.gen_valve_setting:
            featlen_dict[ParamEnum.VALVE_SETTING] = num_valves
        if args.gen_valve_diameter:
            featlen_dict[ParamEnum.VALVE_DIAMETER] = num_valves

    if args.gen_res_total_head and len(wn.reservoirs) > 0:
        featlen_dict[ParamEnum.RESERVOIR_TOTALHEAD] = len(wn.reservoirs)

    print("Start simulation...")
    print("saved_path = ", saved_path)
    skip_nodes = skip_links = []
    num_skip_nodes = num_skip_links = 0
    if config.has_option("general", "skip_nodes"):
        skip_nodes = config.get("general", "skip_nodes").strip().split(",")

    if args.skip_resevoir_result:
        skip_nodes.extend(wn.reservoirs.uid.to_list())

    num_skip_nodes = len(skip_nodes)

    print(f"skip nodes = {skip_nodes}")
    print(f"#skip_nodes = {num_skip_nodes}")

    if config.has_option("general", "skip_links"):
        skip_links = config.get("general", "skip_links").strip().split(",")
        num_skip_links = len(skip_links)
    print(f"#skip_links = {num_skip_links}")

    node_uids = wn.nodes.uid
    num_result_nodes = len(node_uids.loc[~node_uids.isin(skip_nodes)]) if skip_nodes else len(node_uids)
    print(f"exepected #result_nodes = {num_result_nodes} | Note that if attribute is 'demand', #results_nodes should be #junctions")

    link_uids = wn.links.uid
    num_result_links = len(link_uids.loc[~link_uids.isin(skip_links)]) if skip_links else len(link_uids)
    print(f"exepected #result_links = {num_result_links}")
    store = zarr.DirectoryStore(zarr_storage_dir)
    tg = RayTokenGenerator(store=store, num_scenes=backup_num_scenarios, featlen_dict=featlen_dict, num_chunks=num_chunks)

    tg.sequential_update(args=args)
    ragged_tokens = tg.load_computed_params()
    root_group = zarr.open_group(store, synchronizer=zarr.ThreadSynchronizer())

    tmp_group = root_group.create_group("tmp", overwrite=True)
    for att in expected_attributes:
        if att in support_node_attr_keys:
            if att == "demand":
                uids = wn.junctions.uid
                num_junctions = len(uids.loc[~uids.isin(skip_nodes)]) if skip_nodes else len(uids)
                tmp_group.create(att, shape=[num_scenarios, num_junctions], chunks=[batch_size, num_result_nodes], overwrite=True)
            else:
                tmp_group.create(att, shape=[num_scenarios, num_result_nodes], chunks=[batch_size, num_result_nodes], overwrite=True)

        elif att in support_link_attr_keys:
            tmp_group.create(att, shape=[num_scenarios, num_result_links], chunks=[batch_size, num_result_links], overwrite=True)

    try:
        sim_start = time()

        token_ids = []
        scene_ids = []

        for batch_id in range(num_batches):
            start_id = batch_id * batch_size
            end_id = start_id + batch_size
            batch_ragged_tokens = ragged_tokens[start_id:end_id]
            token_ids.append(ray.put(batch_ragged_tokens))
            scene_ids.append(ray.put([start_id + x for x in range(batch_size)]))

        executors = [
            WDNRayExecutor.remote(
                featlen_dict=featlen_dict,
                config=config,
                valve_type_dict=valve_type_dict,
                args=args,
            )
            for _ in range(num_executors)
        ]

        start_index = 0
        progressbar = tqdm(total=num_batches)
        result_worker_dict = {e.simulate.remote(token_ids.pop(), scene_ids.pop()): e for e in executors if scene_ids}
        done_ids, _ = ray.wait(list(result_worker_dict), num_returns=1)

        ordered_names_dict = {}
        success_scenarios = 0
        while done_ids and success_scenarios < num_scenarios:
            done_worker_id = done_ids[0]
            catch_error = False
            try:
                result, ordered_name_list = ray.get(done_worker_id)
            except RayError as e:
                print(f"WARNING! Ray error {e}")
                catch_error = True
            worker = result_worker_dict.pop(done_worker_id)
            if scene_ids:
                result_worker_dict[worker.simulate.remote(token_ids.pop(), scene_ids.pop())] = worker

            if not catch_error:
                success_size = 0
                for key, value in result.items():
                    if key not in ordered_names_dict:
                        ordered_names_dict[key] = ordered_name_list

                    if start_index + value.shape[0] < tmp_group[key].shape[0]:
                        success_size = value.shape[0]
                        tmp_group[key][start_index : start_index + success_size] = value
                    else:
                        success_size = tmp_group[key].shape[0] - start_index
                        tmp_group[key][start_index : start_index + success_size] = value[:success_size]

                del result
                start_index += success_size
                success_scenarios += success_size
            progressbar.update(1)
            done_ids, _ = ray.wait(list(result_worker_dict), num_returns=1)

        progressbar.close()
        ray.shutdown()

        elapsed_time = time() - sim_start
        print(f"\nSimulation time: {elapsed_time} seconds")
        print(f"Process run on {num_batches} batches, total scenes: {backup_num_scenarios}")
        print(f"Success/Expected: {success_scenarios}/{num_scenarios} scenes")

        del root_group[ParamEnum.RANDOM_TOKEN]

        indent = 16
        precision = 8
        if success_scenarios > 0:
            for name in list(tmp_group.keys()):
                root_group.create_group(name, overwrite=True)
                if success_scenarios != num_scenarios:
                    tmp_group[name].resize(success_scenarios, tmp_group[name].shape[-1])

            train_index = int(success_scenarios * train_ratio)
            valid_index = train_index + int(success_scenarios * valid_ratio)

            key_list = list(tmp_group.keys())

            config_dict = {sect: dict(config.items(sect)) for sect in config.sections()}
            if skip_nodes:
                config_dict["skip_nodes"] = skip_nodes
            if skip_links:
                config_dict["skip_links"] = skip_links

            root_group.attrs["config"] = config_dict
            root_group.attrs["args"] = vars(args)
            root_group.attrs["ordered_names_by_attr"] = ordered_names_dict

            for key in key_list:
                a = tmp_group[key]
                train_a, valid_a, test_a = a[:train_index], a[train_index:valid_index], a[valid_index:]

                train_a_df = pd.DataFrame(train_a).astype(float)
                train_min = train_a.min()
                train_max = train_a.max()
                train_mean = train_a.mean()
                train_std = train_a.std()

                train_mean_feat_coef = train_a_df.corr().mean().mean()  # np.corrcoef(train_a.T).mean()
                train_mean_batch_coef = train_a_df.T.corr().mean().mean()  # np.corrcoef(train_a).mean()
                train_cv = (train_a.var(axis=-1) / train_a.mean(axis=-1)).mean()

                root_group[key].attrs["min"] = train_min
                root_group[key].attrs["max"] = train_max
                root_group[key].attrs["mean"] = train_mean
                root_group[key].attrs["std"] = train_std
                root_group[key].attrs["mcoef"] = train_mean_feat_coef
                root_group[key].attrs["bcoef"] = train_mean_batch_coef
                root_group[key].attrs["cv"] = train_cv

                print(f"##############################{key}###############################################")

                print(f"Mean:   {train_mean:>{indent}.{precision}f}")
                print(f"Std:    {train_std:>{indent}.{precision}f}")
                print(f"Min:    {train_min:>{indent}.{precision}f}")
                print(f"Max:    {train_max:>{indent}.{precision}f}")
                print(f"CV:     {train_cv:>{indent}.{precision}f}")
                print(f"FCoef:  {train_mean_feat_coef:>{indent}.{precision}f}")
                print(f"BCoef:  {train_mean_batch_coef:>{indent}.{precision}f}")

                key_train = os.path.join(key, "train")
                root_group.empty_like(key_train, train_a, chunks=(batch_size, a.chunks[-1]))
                root_group[key_train][:] = train_a
                print(f"\n{key_train}.info: {root_group[key_train].info}")

                key_valid = os.path.join(key, "valid")
                root_group.empty_like(key_valid, valid_a, chunks=(batch_size, a.chunks[-1]))
                root_group[key_valid][:] = valid_a
                print(f"\n{key_valid}.info: {root_group[key_valid].info}")

                key_test = os.path.join(key, "test")
                root_group.empty_like(key_test, test_a, chunks=(batch_size, a.chunks[-1]))
                root_group[key_test][:] = test_a
                print(f"\n{key_test}.info: {root_group[key_test].info}")

            del root_group["tmp"]

            elapsed_time = time() - program_start
            print(f"\nExecution time: {elapsed_time} seconds")

            store2 = zarr.ZipStore(saved_path + ".zip", mode="w")
            zarr.copy_store(store, store2, if_exists="replace")
            store2.close()
            print(root_group.tree())
    except Exception as e:
        print(e)


def main():
    generate_dataset(CONFIG)


if __name__ == "__main__":
    main()
