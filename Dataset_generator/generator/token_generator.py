from configparser import ConfigParser
import numpy as np
from enum import Enum
from epynet import Network, ObjectCollection
from epynet import epanet2
from . import epynet_utils as eutils
import os
import ray
import zarr
from collections import defaultdict
from numcodecs import Blosc


EPSILON = 1e-12


class ParamEnum(str, Enum):
    RANDOM_TOKEN        = "token"
    JUNC_DEMAND         = "junc_demand"
    PUMP_STATUS         = "pump_status"
    PUMP_SPEED          = "pump_speed"
    TANK_LEVEL          = "tank_level"
    VALVE_SETTING       = "valve_setting"
    VALVE_STATUS        = "valve_status"
    PIPE_ROUGHNESS      = "pipe_roughness"
    RESERVOIR_TOTALHEAD = "reservoir_totalhead"


def compute_contineous_values_by_range(tokens, ratios, ori_vals=None, **kwargs):
    range_lo, range_hi = ratios[0], ratios[1]
    return range_lo + tokens * (range_hi - range_lo)


def compute_boolean_values(tokens, ratios, **kwargs):
    open_prob = ratios[0]
    return np.less(tokens, open_prob).astype(tokens.dtype)


def generate_params(tokens, ratios, target_object_collection, get_original_values_fn, update_formula_fn, **kwargs):
    ori_vals = np.array(list(map(get_original_values_fn, target_object_collection)))
    if sum(ratios) == 0.0:
        return ori_vals
    return update_formula_fn(tokens=tokens, ratios=ratios, ori_vals=ori_vals, **kwargs)


@ray.remote
def ray_batch_update(chunk_size, num_features, featlen_dict, args):
    return batch_update(chunk_size, num_features, featlen_dict, args)


def batch_update(chunk_size, num_features, featlen_dict, args):
    tokens = np.random.uniform(low=0.0, high=1.0, size=(chunk_size, num_features))

    config = ConfigParser()
    config.read(args.config)
    config_keys = dict(config.items()).keys()

    wn_inp_path = config.get("general", "wn_inp_path")
    wn = Network(wn_inp_path)

    ragged_tokens = eutils.RaggedArrayDict.from_keylen_and_stackedarray(featlen_dict, tokens)

    new_tokens = defaultdict()

    if "junction" in config_keys:
        if args.gen_demand:
            def get_origin_dmd(junc):
                return junc.basedemand * junc.pattern.values[0] if eutils.ENhasppatern(junc) else 0.0
            new_tokens[ParamEnum.JUNC_DEMAND] = generate_params(
                tokens=ragged_tokens[ParamEnum.JUNC_DEMAND],
                ratios=[config.getfloat("junction", "demand_lo"), config.getfloat("junction", "demand_hi")],
                target_object_collection=wn.junctions,
                get_original_values_fn=get_origin_dmd,
                update_formula_fn=compute_contineous_values_by_range,
            )

    if "pump" in config_keys:
        if args.gen_pump_init_status:
            new_tokens[ParamEnum.PUMP_STATUS] = generate_params(
                tokens=ragged_tokens[ParamEnum.PUMP_STATUS],
                ratios=[config.getfloat("pump", "open_prob")],
                target_object_collection=wn.pumps,
                get_original_values_fn=lambda pump: pump.initstatus,
                update_formula_fn=compute_boolean_values,
            )
        if args.gen_pump_speed:
            new_tokens[ParamEnum.PUMP_SPEED] = generate_params(
                tokens=ragged_tokens[ParamEnum.PUMP_SPEED],
                ratios=[config.getfloat("pump", "speed_lo"), config.getfloat("pump", "speed_hi")],
                target_object_collection=wn.pumps,
                get_original_values_fn=lambda pump: pump.speed,
                update_formula_fn=compute_contineous_values_by_range,
            )

    if "tank" in config_keys:
        if args.gen_tank_level:
            new_tokens[ParamEnum.TANK_LEVEL] = generate_params(
                tokens=ragged_tokens[ParamEnum.TANK_LEVEL],
                ratios=[config.getfloat("tank", "level_lo"), config.getfloat("tank", "level_hi")],
                target_object_collection=wn.tanks,
                get_original_values_fn=lambda tank: tank.tanklevel,
                update_formula_fn=compute_contineous_values_by_range,
            )

    if "valve" in config_keys:
        if args.gen_valve_setting:
            valve_type_ratio_dict = {}
            valve_type_uid_dict = {}
            for v in wn.valves:
                if v.valve_type not in valve_type_ratio_dict:
                    key = v.valve_type.lower()
                    ratio_lo = config.getfloat("valve", f"setting_{key}_lo")
                    ratio_hi = config.getfloat("valve", f"setting_{key}_hi")
                    valve_type_ratio_dict[v.valve_type] = (ratio_lo, ratio_hi)
                    valve_type_uid_dict[v.valve_type] = []
                valve_type_uid_dict[v.valve_type].append(v.uid)
            overridden_values = np.zeros(shape=[chunk_size, len(wn.valves)])
            for valve_type in valve_type_ratio_dict:
                ratios = valve_type_ratio_dict[valve_type]
                uids = valve_type_uid_dict[valve_type]
                target_object_collection = ObjectCollection({k: wn.valves[k] for k in uids if k in wn.valves})
                in_uids_mask = np.isin(list(wn.valves.keys()), uids)
                valve_type_new_tokens = generate_params(
                    tokens=ragged_tokens[ParamEnum.VALVE_SETTING][:, in_uids_mask],
                    ratios=ratios,
                    target_object_collection=target_object_collection,
                    get_original_values_fn=lambda v: v.setting,
                    update_formula_fn=compute_contineous_values_by_range,
                )
                overridden_values[:, in_uids_mask] = valve_type_new_tokens
            new_tokens[ParamEnum.VALVE_SETTING] = overridden_values

        if args.gen_valve_init_status:
            new_tokens[ParamEnum.VALVE_STATUS] = generate_params(
                tokens=ragged_tokens[ParamEnum.VALVE_STATUS],
                ratios=[config.getfloat("valve", "open_prob")],
                target_object_collection=wn.valves,
                get_original_values_fn=lambda v: v.initstatus,
                update_formula_fn=compute_boolean_values,
            )

    if "pipe" in config_keys:
        if args.gen_roughness:
            new_tokens[ParamEnum.PIPE_ROUGHNESS] = generate_params(
                tokens=ragged_tokens[ParamEnum.PIPE_ROUGHNESS],
                ratios=[config.getfloat("pipe", "roughness_lo"), config.getfloat("pipe", "roughness_hi")],
                target_object_collection=wn.pipes,
                get_original_values_fn=lambda p: p.roughness,
                update_formula_fn=compute_contineous_values_by_range,
            )

    if "reservoir" in config_keys:
        if args.gen_res_total_head:
            def get_original_res_head(res):
                base_head = res.elevation
                try:
                    p_index = res.get_object_value(epanet2.EN_PATTERN)
                    head = wn.ep.ENgetpatternvalue(int(p_index), 1)
                except epanet2.ENtoolkitError:
                    head = 1.0
                return base_head * head

            new_tokens[ParamEnum.RESERVOIR_TOTALHEAD] = generate_params(
                tokens=ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD],
                ratios=[config.getfloat("reservoir", "head_lo"), config.getfloat("reservoir", "head_hi")],
                target_object_collection=wn.reservoirs,
                get_original_values_fn=get_original_res_head,
                update_formula_fn=compute_contineous_values_by_range,
            )

    concated_arrays = [new_tokens[k] for k in featlen_dict.keys()]
    return np.concatenate(concated_arrays, axis=-1)


class RayTokenGenerator:
    def __init__(self, store, num_scenes, featlen_dict, num_chunks):
        self.store = store
        self.num_scenes = num_scenes
        self.featlen_dict = featlen_dict
        self.num_chunks = num_chunks
        self.num_features = sum(self.featlen_dict.values())

    def sequential_update(self, args):
        chunk_size = args.batch_size
        num_chunks = self.num_scenes // chunk_size
        num_out_features = sum(self.featlen_dict.values())
        start_index = 0
        from tqdm import tqdm
        progressbar = tqdm(total=num_chunks)
        for _ in range(num_chunks):
            result = batch_update(chunk_size, self.num_features, self.featlen_dict, args)
            if start_index == 0:
                z_tokens = zarr.empty(
                    [self.num_scenes, num_out_features],
                    chunks=(chunk_size, num_out_features),
                    dtype="f8",
                    store=os.path.join(self.store.path, ParamEnum.RANDOM_TOKEN),
                    overwrite=True,
                    synchronizer=zarr.ThreadSynchronizer(),
                    compressor=Blosc(cname="lz4", clevel=5),
                )
            z_tokens[start_index : start_index + chunk_size] = result
            start_index += chunk_size
            del result
            progressbar.update(1)
        progressbar.close()
        print("OK")

    def load_computed_params(self):
        return zarr.open_array(store=os.path.join(self.store.path, ParamEnum.RANDOM_TOKEN), mode="r")
