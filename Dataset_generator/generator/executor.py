import glob
import os
import ray
from epynet import Network
from copy import deepcopy
import numpy as np
from epynet import epanet2
import networkx as nx
from . import epynet_utils as eutils
from .token_generator import ParamEnum


class WDNExecutor(object):
    def __init__(self, featlen_dict, config, valve_type_dict, args, wn=None):
        self.sort_node_name = False
        self.min_valve_setting = 1e-4
        self.ordered = False
        self.custom_base_index = 100

        self.featlen_dict = deepcopy(featlen_dict)
        self.config = config

        self.expected_attr = args.att.strip().split(",")
        self.pressure_upperbound = args.pressure_upperbound
        self.pressure_lowerbound = args.pressure_lowerbound
        self.init_valve_state = args.init_valve_state
        self.init_pipe_state = args.init_pipe_state

        self.gen_demand = args.gen_demand
        self.gen_pipe_roughness = args.gen_roughness
        self.gen_valve_init_status = args.gen_valve_init_status
        self.gen_valve_setting = args.gen_valve_setting
        self.gen_pump_init_status = args.gen_pump_init_status
        self.gen_pump_speed = args.gen_pump_speed
        self.gen_tank_level = args.gen_tank_level
        self.gen_res_total_head = args.gen_res_total_head

        self.replace_nonzero_basedmd = args.replace_nonzero_basedmd
        self.remove_pattern = args.remove_pattern
        self.convert_results_by_flow_unit = args.convert_results_by_flow_unit

        wn_inp_path = self.config.get("general", "wn_inp_path")
        self.skip_nodes = self.config.get("general", "skip_nodes").strip().split(",") if self.config.has_option("general", "skip_nodes") else []
        self.skip_links = self.config.get("general", "skip_links").strip().split(",") if self.config.has_option("general", "skip_links") else []

        if wn is not None:
            self.wn = wn
        else:
            self.wn = Network(wn_inp_path)

        if self.remove_pattern:
            patterns = self.wn.patterns
            if len(patterns) > 0:
                for p in patterns:
                    eutils.ENdeletepattern(self.wn, p.uid)

        patterns = self.wn.patterns
        for i, _ in enumerate(self.wn.junctions):
            if not str(self.custom_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_base_index + i), values=[0])

        self.custom_res_pattern_base_index = self.custom_base_index + len(self.wn.junctions)
        for i, _ in enumerate(self.wn.reservoirs):
            if not str(self.custom_res_pattern_base_index + i) in patterns:
                self.wn.add_pattern(str(self.custom_res_pattern_base_index + i), values=[0])

    def filter_skip_elements(self, df, skip_list):
        mask = df.index.isin(skip_list)
        return df.loc[np.invert(mask)]

    def epynet_simulate2(self, tokens, scene_id):
        self.wn.reset()

        ragged_tokens = eutils.RaggedArrayDict.from_keylen_and_stackedarray(self.featlen_dict, tokens, axis=0)

        junc_demands  = ragged_tokens[ParamEnum.JUNC_DEMAND]
        pump_statuses = ragged_tokens[ParamEnum.PUMP_STATUS]
        pump_speed    = ragged_tokens[ParamEnum.PUMP_SPEED]
        pipe_roughness = ragged_tokens[ParamEnum.PIPE_ROUGHNESS]
        tank_levels   = ragged_tokens[ParamEnum.TANK_LEVEL]
        valve_statuses = ragged_tokens[ParamEnum.VALVE_STATUS]
        valve_settings = ragged_tokens[ParamEnum.VALVE_SETTING]
        res_heads     = ragged_tokens[ParamEnum.RESERVOIR_TOTALHEAD]

        self.wn.ep.ENsettimeparam(epanet2.EN_DURATION, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_QUALSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_PATTERNSTART, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTEP, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_REPORTSTART, 1)
        self.wn.ep.ENsettimeparam(epanet2.EN_RULESTEP, 1)

        support_node_attr_keys = ["demand", "head", "pressure"]
        support_link_attr_keys = ["velocity", "flow"]

        for i, junc in enumerate(self.wn.junctions):
            if self.gen_demand:
                junc.basedemand = 1.0
                junc.pattern = str(self.custom_base_index + i)
                junc.pattern.values = [junc_demands[i]]
                eutils.ENsetdemandpatterntoallcategories(self.wn, junc.index, junc.basedemand, junc.pattern.index)

        for i, pump in enumerate(self.wn.pumps):
            if self.gen_pump_init_status:
                pump.initstatus = int(pump_statuses[i])
            if self.gen_pump_speed:
                pump.speed = pump_speed[i]

        for i, tank in enumerate(self.wn.tanks):
            if self.gen_tank_level:
                tank_level = tank.minlevel + tank_levels[i] * (tank.maxlevel - tank.minlevel)
                eutils.set_object_value_wo_ierror(obj=tank, code=epanet2.EN_TANKLEVEL, value=tank_level)

        tmp_graph = eutils.get_networkx_graph(wn=self.wn, include_reservoir=True, graph_type="undirected")
        for i, valve in enumerate(self.wn.valves):
            if self.init_valve_state is not None:
                valve.initstatus = int(self.init_valve_state)

            if self.gen_valve_init_status:
                if not bool(valve_statuses[i]):
                    tmp_graph.remove_edge(valve.from_node.uid, valve.to_node.uid)
                    if nx.is_connected(tmp_graph):
                        valve.initstatus = int(valve_statuses[i])
                    else:
                        tmp_graph.add_edge(valve.from_node.uid, valve.to_node.uid)
                        valve.initstatus = True
                else:
                    valve.initstatus = int(valve_statuses[i])

            if self.gen_valve_setting:
                if valve_settings[i] > 0:
                    eutils.set_object_value_wo_ierror(obj=valve, code=epanet2.EN_INITSETTING, value=valve_settings[i])

        for i, pipe in enumerate(self.wn.pipes):
            if self.init_pipe_state is not None and not pipe.check_valve:
                eutils.set_object_value_wo_ierror(obj=pipe, code=epanet2.EN_INITSTATUS, value=int(self.init_pipe_state))
            if self.gen_pipe_roughness:
                pipe.roughness = pipe_roughness[i]

        if self.gen_res_total_head:
            for i, res in enumerate(self.wn.reservoirs):
                res.set_object_value(epanet2.EN_ELEVATION, 1.0)
                tmp = res_heads[i]
                p_index = self.wn.ep.ENgetpatternindex(str(self.custom_res_pattern_base_index + i))
                self.wn.ep.ENsetpattern(p_index, [tmp])
                res.set_object_value(epanet2.EN_PATTERN, p_index)

        sim_results = {}
        prefix_name = "tmp_" + str(scene_id)
        for file in glob.glob(f"{prefix_name}.*"):
            os.remove(file)

        def ENrunH(ep):
            ierr = ep._lib.EN_runH(ep.ph, epanet2.ctypes.byref(ep._current_simulation_time))
            return ierr

        def solve_return_error(wn, simtime=0):
            if wn.solved and wn.solved_for_simtime == simtime:
                return
            wn.ep.ENsettimeparam(4, simtime)
            wn.ep.ENopenH()
            wn.ep.ENinitH(0)
            code = ENrunH(wn.ep)
            assert code is not None
            wn.ep.ENcloseH()
            wn.solved = True
            wn.solved_for_simtime = simtime
            return code

        code = solve_return_error(self.wn)

        if self.skip_nodes is not None:
            pressure_df = self.wn.nodes.pressure
            pressure_results = self.filter_skip_elements(pressure_df, self.skip_nodes).values
            pressure_results = np.reshape(pressure_results, [1, -1])
        else:
            pressure_results = np.reshape(self.wn.nodes.pressure.values, [1, -1])

        if self.convert_results_by_flow_unit is not None:
            from_unit = eutils.FlowUnits[self.wn.ep.ENgetflowunits()]
            to_unit = self.convert_results_by_flow_unit
            if from_unit != to_unit:
                pressure_results = eutils.ENconvert(from_unit=from_unit, to_unit=to_unit, hydraulic_param="pressure", values=pressure_results)

        error = np.isnan(pressure_results).any()
        if code is not None and code > 0:
            error = error or code > 0

        if self.pressure_lowerbound is not None:
            error = error or any(pressure_results.min(axis=1) < self.pressure_lowerbound)

        if self.pressure_upperbound is not None:
            error = error or any(pressure_results.max(axis=1) > self.pressure_upperbound)

        sim_result_indices = None
        for attr in self.expected_attr:
            if attr in support_node_attr_keys:
                sim_result = getattr(self.wn.nodes, attr) if hasattr(self.wn.nodes, attr) else getattr(self.wn.junctions, attr)
                if self.skip_nodes is not None:
                    sim_result = self.filter_skip_elements(sim_result, self.skip_nodes)
            elif attr in support_link_attr_keys:
                sim_result = getattr(self.wn.links, attr)
                if self.skip_links is not None:
                    sim_result = self.filter_skip_elements(sim_result, self.skip_links)

            if self.sort_node_name:
                sim_result = sim_result.sort_index(axis=1)

            sim_result_indices = sim_result.index.tolist()
            sim_result = np.reshape(sim_result.to_numpy(), [1, -1])
            if self.convert_results_by_flow_unit is not None:
                from_unit = eutils.FlowUnits[self.wn.ep.ENgetflowunits()]
                to_unit = self.convert_results_by_flow_unit
                if from_unit != to_unit:
                    sim_result = eutils.ENconvert(from_unit=from_unit, to_unit=to_unit, hydraulic_param=attr, values=sim_result)
            sim_results[attr] = sim_result

        return sim_results, error, sim_result_indices

    def update_batch_dict(self, batch_dict, single_dict):
        for key, value in single_dict.items():
            if key not in batch_dict:
                batch_dict[key] = value
            else:
                batch_dict[key] = np.concatenate([batch_dict[key], value], axis=0)
        return batch_dict

    def check_order(self, l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    def simulate(self, batch_tokens, scence_ids):
        batch_results = {}
        batch_size = batch_tokens.shape[0]
        stored_ordered_name_list = None
        for id in range(batch_size):
            tokens = batch_tokens[id]
            single_result, error, ordered_name_list = self.epynet_simulate2(tokens, scence_ids[id])

            if stored_ordered_name_list is not None:
                assert self.check_order(ordered_name_list, stored_ordered_name_list)
            stored_ordered_name_list = ordered_name_list

            if not error:
                batch_results = self.update_batch_dict(batch_results, single_result)

        return batch_results, stored_ordered_name_list


@ray.remote(num_cpus=0)
class WDNRayExecutor(WDNExecutor):
    pass
