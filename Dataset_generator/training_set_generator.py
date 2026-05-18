import sys
import os
import argparse
import shutil
import zarr
from configparser import ConfigParser
from time import time
from tqdm import tqdm
from epynet import Network
import ray
import pandas as pd
from ray.exceptions import RayError

from generator.token_generator import ParamEnum, RayTokenGenerator
from generator.executor import WDNRayExecutor


_CONFIG      = "configs/v7.1/ctown_config.ini"
_BATCH_SIZE  = 50
_EXECUTORS   = 8
_ATT         = "pressure,head"
_TRAIN_RATIO = 0.6
_VALID_RATIO = 0.2


def get_arguments(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=_CONFIG,     type=str)
    parser.add_argument("--batch_size", default=_BATCH_SIZE, type=int)
    parser.add_argument("--executors",  default=_EXECUTORS,  type=int)
    parser.add_argument("--att",         default=_ATT,         type=str)
    parser.add_argument("--train_ratio", default=_TRAIN_RATIO, type=float)
    parser.add_argument("--valid_ratio", default=_VALID_RATIO, type=float)

    parser.add_argument("--gen_demand",          default=False, action="store_true")
    parser.add_argument("--gen_res_total_head",  default=False, action="store_true")
    parser.add_argument("--gen_tank_level",      default=False, action="store_true")
    parser.add_argument("--gen_pump_init_status",default=False, action="store_true")
    parser.add_argument("--gen_pump_speed",      default=False, action="store_true")
    parser.add_argument("--gen_roughness",       default=False, action="store_true")
    parser.add_argument("--gen_valve_setting",   default=False, action="store_true")
    parser.add_argument("--gen_valve_init_status",default=False, action="store_true")

    parser.add_argument("--pressure_lowerbound", default=None, type=float)
    parser.add_argument("--pressure_upperbound", default=None, type=float)

    args = parser.parse_args(args=raw_args)

    # Fixed internal defaults (not exposed as CLI)
    args.remove_pattern           = True
    args.init_valve_state         = 1
    args.init_pipe_state          = None
    args.replace_nonzero_basedmd  = False
    args.update_totalhead_method  = None
    args.accept_warning_code      = False
    args.convert_results_by_flow_unit = "LPS"
    args.allow_error              = False
    args.skip_resevoir_result     = False
    args.debug                    = False
    args.flowrate_threshold       = None

    return args


def generate_dataset(args):
    program_start = time()

    config = ConfigParser()
    config.read(args.config)

    wn_inp_path = config.get("general", "wn_inp_path")
    storage_dir = config.get("general", "storage_dir")

    zarr_storage_dir = os.path.join(storage_dir, "zarrays")
    os.makedirs(storage_dir, exist_ok=True)
    shutil.rmtree(path=storage_dir, ignore_errors=True)
    os.makedirs(zarr_storage_dir, exist_ok=False)

    saved_path        = storage_dir
    num_scenarios     = config.getint("general", "num_scenarios")
    backup_num_scenarios = num_scenarios * 10
    batch_size        = args.batch_size
    num_executors     = args.executors
    expected_attributes = args.att.strip().split(",")
    train_ratio       = args.train_ratio
    valid_ratio       = args.valid_ratio
    num_batches       = backup_num_scenarios // batch_size
    num_chunks        = backup_num_scenarios // batch_size

    support_node_attr_keys = ["head", "pressure", "demand"]
    support_link_attr_keys = ["flow", "velocity"]
    support_keys = list(set(support_node_attr_keys).union(support_link_attr_keys))
    for a in expected_attributes:
        if a not in support_keys:
            raise AttributeError(f"{a} is not found or not supported!")

    wn = Network(wn_inp_path)
    valve_type_dict = {}

    featlen_dict = dict()
    if len(wn.junctions) > 0 and args.gen_demand:
        featlen_dict[ParamEnum.JUNC_DEMAND] = len(wn.junctions)
    if len(wn.pipes) > 0 and args.gen_roughness:
        featlen_dict[ParamEnum.PIPE_ROUGHNESS] = len(wn.pipes)
    if len(wn.pumps) > 0:
        if args.gen_pump_init_status:
            featlen_dict[ParamEnum.PUMP_STATUS] = len(wn.pumps)
        if args.gen_pump_speed:
            featlen_dict[ParamEnum.PUMP_SPEED] = len(wn.pumps)
    if len(wn.tanks) > 0 and args.gen_tank_level:
        featlen_dict[ParamEnum.TANK_LEVEL] = len(wn.tanks)
    if len(wn.valves) > 0:
        if args.gen_valve_init_status:
            featlen_dict[ParamEnum.VALVE_STATUS] = len(wn.valves)
        if args.gen_valve_setting:
            featlen_dict[ParamEnum.VALVE_SETTING] = len(wn.valves)
    if args.gen_res_total_head and len(wn.reservoirs) > 0:
        featlen_dict[ParamEnum.RESERVOIR_TOTALHEAD] = len(wn.reservoirs)

    print("Start simulation...")
    print("saved_path = ", saved_path)
    skip_nodes = skip_links = []
    if config.has_option("general", "skip_nodes"):
        skip_nodes = config.get("general", "skip_nodes").strip().split(",")
    num_skip_nodes = len(skip_nodes)
    print(f"skip nodes = {skip_nodes}")
    print(f"#skip_nodes = {num_skip_nodes}")
    if config.has_option("general", "skip_links"):
        skip_links = config.get("general", "skip_links").strip().split(",")
    print(f"#skip_links = {len(skip_links)}")

    node_uids = wn.nodes.uid
    num_result_nodes = len(node_uids.loc[~node_uids.isin(skip_nodes)]) if skip_nodes else len(node_uids)
    print(f"expected #result_nodes = {num_result_nodes}")

    link_uids = wn.links.uid
    num_result_links = len(link_uids.loc[~link_uids.isin(skip_links)]) if skip_links else len(link_uids)
    print(f"expected #result_links = {num_result_links}")

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
            token_ids.append(ray.put(ragged_tokens[start_id:end_id]))
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
                train_min  = train_a.min()
                train_max  = train_a.max()
                train_mean = train_a.mean()
                train_std  = train_a.std()
                train_mean_feat_coef  = train_a_df.corr().mean().mean()
                train_mean_batch_coef = train_a_df.T.corr().mean().mean()
                train_cv   = (train_a.var(axis=-1) / train_a.mean(axis=-1)).mean()

                root_group[key].attrs["min"]    = train_min
                root_group[key].attrs["max"]    = train_max
                root_group[key].attrs["mean"]   = train_mean
                root_group[key].attrs["std"]    = train_std
                root_group[key].attrs["mcoef"]  = train_mean_feat_coef
                root_group[key].attrs["bcoef"]  = train_mean_batch_coef
                root_group[key].attrs["cv"]     = train_cv

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


if __name__ == "__main__":
    args = get_arguments(sys.argv[1:])
    generate_dataset(args)
