from __future__ import annotations
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_utils.slicing import slice_by_depth, slice_by_gate_types, slice_by_coloring, combine_slices
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.simplify import OperatorBudget
from collections.abc import Sequence
import rustworkx as rx



def auto_color_edges(edges: Sequence[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Color the input edges of an undirected graph such that no two incident edges share a color.

    Args:
        edges: The edges describing an undirected graph.

    Returns:
        A dictionary mapping each edge to an integer representation of a color.
    """
    coupling_graph: rx.PyGraph = rx.PyGraph()
    coupling_graph.extend_from_edge_list(sorted(edges, key=lambda x: min(x)))
    edge_coloring_by_id = rx.graph_greedy_edge_color(coupling_graph)

    coloring_out = {}
    for i, edge in enumerate(coupling_graph.edge_list()):
        coloring_out[edge] = edge_coloring_by_id[i]

    # This function should always return a color for each unique input edge
    assert len(coloring_out.keys()) == len(set(edges))

    return coloring_out

def extract_edges(circuit):
    edges = []
    for inst in circuit.data:
        if len(inst.qubits) == 2:
            q1 = inst.qubits[0]._index
            q2 = inst.qubits[1]._index
            if (q1,q2) not in edges and (q2,q1) not in edges:
                edges.append((q1,q2))
    return edges

"""
def extract_edges(circuit: QuantumCircuit) -> list[tuple[int, int]]:
  edges = []
 
  for inst in circuit.data:

    if len(inst.qubits) == 2:
        q1 = circuit.qubits.index(inst.qubits[0])
        q2 = circuit.qubits.index(inst.qubits[1])
        edge=tuple(sorted((q1,q2)))
        edges.append((q1,q2))

  return edges
"""

def do_backpropagation(circuit: QuantumCircuit, observable: SparsePauliOp,
                       op_budget: OperatorBudget, backend)-> int:
    #circuit = circuit.decompose(reps=10) # the circuit should be transpiled circuit..do it.
    if circuit.num_clbits > 0: # circuit has measurement
        circuit.remove_final_measurements(inplace=True)
        if circuit.num_clbits > 0: # implies mid circuit measurements
            return [None,None, None], [None, None, None]
    depth_val=[]
    num_slices = []
    slices = [slice_by_depth(circuit, max_slice_depth=1), slice_by_gate_types(circuit), slice_by_coloring(circuit, auto_color_edges(extract_edges(circuit)))]
    #num_slices = [len(slice) for slice in slices if type(len(slices))==int else None]
    for idx, slice in enumerate(slices):
        if type(len(slice))==int:
            num_slices.append(len(slice))
        else:
            num_slices.append(None)
        #print("This {idx} slice is starting")
        bp_obs, remaining_slices, metadata = backpropagate(observable, slice , operator_budget=op_budget, max_seconds=180)

        bp_circuit = combine_slices(remaining_slices, include_barriers=True)
        if type(bp_circuit) != QuantumCircuit:
            #print("Not a qc")
            depth_val.append(None)
        else:
            depth=bp_circuit.depth()
            #print(depth)
            depth_val.append(depth)


    return num_slices, depth_val

 
