import numpy as np
import networkx as nx
from epynet import epanet2
import epynet
import pint


def get_networkx_graph(wn, include_reservoir=True, graph_type="multi_directed"):
    if graph_type == "undirected":
        G = nx.Graph()
    elif graph_type == "directed":
        G = nx.DiGraph()
    elif graph_type == "multi_undirected":
        G = nx.MultiGraph()
    elif graph_type == "multi_directed":
        G = nx.MultiDiGraph()
    else:
        raise NotImplementedError()

    node_list = []
    collection = wn.junctions if not include_reservoir else wn.nodes
    for node in collection:
        node_list.append(node.uid)

    for pipe in wn.pipes:
        if (pipe.from_node.uid in node_list) and (pipe.to_node.uid in node_list):
            G.add_edge(pipe.from_node.uid, pipe.to_node.uid, weight=1.0, length=pipe.length)
    for pump in wn.pumps:
        if (pump.from_node.uid in node_list) and (pump.to_node.uid in node_list):
            G.add_edge(pump.from_node.uid, pump.to_node.uid, weight=1.0, length=0.0)
    for valve in wn.valves:
        if (valve.from_node.uid in node_list) and (valve.to_node.uid in node_list):
            G.add_edge(valve.from_node.uid, valve.to_node.uid, weight=1.0, length=0.0)

    return G


def set_object_value_wo_ierror(obj, code, value):
    assert hasattr(obj, "_values") and hasattr(obj, "index") and obj.network() is not None
    obj.network().solved = False
    obj._values[code] = value
    try:
        if isinstance(obj, epynet.Node):
            ENsetnodevalue2(obj.network().ep, obj.index, code, value)
        else:
            ENsetlinkvalue2(obj.network().ep, obj.index, code, value)
    except Exception as e:
        print(f"ERROR AT OBJ = {obj.uid}, code = {code}, value = {value}")
        raise Exception(e)


def ENhasppatern(obj):
    if not isinstance(obj, epynet.Junction):
        return False
    try:
        p = obj.pattern
        return p is not None
    except Exception:
        return False


def ENsetnodevalue2(ep, index, paramcode, value):
    ierr = ep._lib.EN_setnodevalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode), epanet2.ctypes.c_float(value))
    if ierr != 0:
        raise Exception(ierr)
    del ierr


def ENsetlinkvalue2(ep, index, paramcode, value):
    ierr = ep._lib.EN_setlinkvalue(ep.ph, epanet2.ctypes.c_int(index), epanet2.ctypes.c_int(paramcode), epanet2.ctypes.c_float(value))
    if ierr != 0:
        raise Exception(ierr)
    del ierr


def ENdeletepattern(wn, pattern_uid, delete_pattern_in_rules=True):
    patten_index = epanet2.ctypes.c_int()
    wn.ep._lib.EN_getpatternindex(wn.ep.ph, epanet2.ctypes.c_char_p(pattern_uid.encode(wn.ep.charset)), epanet2.ctypes.byref(patten_index))
    wn.ep._lib.EN_deletepattern(wn.ep.ph, patten_index)
    if delete_pattern_in_rules:
        wn.ep._lib.EN_deleterule(wn.ep.ph, patten_index)


def ENsetdemandpattern(wn, node_index, demand_category, pattern_index):
    wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_int(pattern_index))


def ENsetdemandpatterntoallcategories(wn, node_index, base_demand, pattern_index):
    demand_category = 1
    ierr = wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_int(pattern_index))
    ierr = wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_double(base_demand))
    while ierr == 0:
        demand_category += 1
        ierr = wn.ep._lib.EN_setdemandpattern(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_int(pattern_index))
        if ierr == 0:
            ierr = wn.ep._lib.EN_setbasedemand(wn.ep.ph, epanet2.ctypes.c_int(node_index), epanet2.ctypes.c_int(demand_category), epanet2.ctypes.c_double(base_demand))


def ENconvert(from_unit, to_unit, hydraulic_param, values):
    us_flow_units = ["CFS", "GPM", "MGD", "IMGD", "AFD"]
    si_flow_units = ["LPS", "LPM", "MLD", "CMH", "CMD"]
    supported_flow_units = list(set(us_flow_units).union(si_flow_units))
    assert from_unit in supported_flow_units
    assert to_unit in supported_flow_units
    assert hydraulic_param in ["pressure", "demand", "head", " velocity", "flow"]
    assert isinstance(values, np.ndarray)

    ureg = pint.UnitRegistry()
    ureg.define("GPM = gallon / minute")
    ureg.define("cubic_meter = meter**3")
    ureg.define("CMH = cubic_meter / hour")
    ureg.define("meter_H2O = 100 * centimeter_H2O")
    ureg.define("CFS = cubic_feet / second")
    ureg.define("MGD = 1000000 * gallon / day")
    ureg.define("IMGD = 1000000 * imperial_gallon / day")
    ureg.define("AFD =  acre_feet / day")
    ureg.define("LPS = liter / second = lps")
    ureg.define("LPM =  liter / minute")
    ureg.define("MLD =  1000000 * liter / day")
    ureg.define("CMD =  cubic_meter / day")

    if hydraulic_param in ["demand", "flow"]:
        leg1 = ureg.Quantity(values, from_unit)
    else:
        if (from_unit in us_flow_units and to_unit in us_flow_units) or (from_unit in si_flow_units and to_unit in si_flow_units):
            return values
        if hydraulic_param == "pressure":
            leg1_punit = "psi" if from_unit in us_flow_units else "meter_H2O"
            leg1 = ureg.Quantity(values, leg1_punit)
        elif hydraulic_param == "head":
            leg1_punit = "feet_H2O" if from_unit in us_flow_units else "meter_H2O"
            leg1 = ureg.Quantity(values, leg1_punit)
        elif hydraulic_param == "velocity":
            leg1_punit = "fps" if from_unit in us_flow_units else "mps"
            leg1 = ureg.Quantity(values, leg1_punit)

    if hydraulic_param in ["demand", "flow"]:
        leg2 = leg1.to(to_unit)
    elif hydraulic_param == "pressure":
        leg2_punit = "psi" if to_unit in us_flow_units else "meter_H2O"
        leg2 = leg1.to(leg2_punit)
    elif hydraulic_param == "head":
        leg2_punit = "feet_H2O" if to_unit in us_flow_units else "meter_H2O"
        leg2 = leg1.to(leg2_punit)
    elif hydraulic_param == "velocity":
        leg2_punit = "fps" if to_unit in us_flow_units else "mps"
        leg2 = leg1.to(leg2_punit)

    return leg2.magnitude


FlowUnits = {
    0: "CFS",
    1: "GPM",
    2: "AFD",
    3: "MGD",
    4: "IMGD",
    5: "LPS",
    6: "LPM",
    7: "MLD",
    8: "CMH",
    9: "CMD",
}


class RaggedArrayList(object):
    def stack_ragged(self, array_list, axis=1):
        lengths = [arr.shape[axis] for arr in array_list]
        idx = np.cumsum(lengths[:-1])
        stacked = np.concatenate(array_list, axis)
        return stacked, idx, lengths

    def __init__(self, array_list, axis=1) -> None:
        self.axis = axis
        if array_list:
            self._stacked_array, self._indices, self._lengths = self.stack_ragged(array_list, axis=self.axis)
        else:
            self._stacked_array = None
            self._indices = None
            self._lengths = []

    def __len__(self):
        if self._lengths:
            return self._indices[-1] + self._lengths[-1]
        else:
            return 0

    def __getitem__(self, index):
        assert index < len(self._lengths)
        cur_length = self._lengths[index]
        if index < 0:
            index = len(self._lengths) + index
        if index < len(self._lengths) - 1:
            next_idx = self._indices[index]
            if self.axis == 0:
                return self._stacked_array[next_idx - cur_length : next_idx]
            else:
                return self._stacked_array[:, next_idx - cur_length : next_idx]
        else:
            if self.axis == 0:
                return self._stacked_array[-cur_length:]
            else:
                return self._stacked_array[:, -cur_length:]


class RaggedArrayDict(RaggedArrayList):
    def __init__(self, array_dict, axis=1) -> None:
        if array_dict:
            self._keys = list(array_dict.keys())
            array_list = list(array_dict.values())
            super().__init__(array_list, axis)
        else:
            self._keys = []
            super().__init__(None, axis)

    @staticmethod
    def from_keylen_and_stackedarray(keylen_dict, stacked_array, axis=1):
        lengths = list(keylen_dict.values())
        indices = np.cumsum(lengths[:-1])
        ragged_tokens = np.split(stacked_array, indices, axis=axis) if len(indices) > 0 else [stacked_array]
        feed_dict = {k: ragged_tokens[i] for i, k in enumerate(keylen_dict)}
        return RaggedArrayDict(feed_dict, axis=axis)

    def __getitem__(self, key):
        assert isinstance(key, str)
        if key in self._keys:
            return super().__getitem__(self._keys.index(key))
        else:
            return None
