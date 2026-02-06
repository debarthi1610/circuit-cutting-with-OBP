from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp

from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.simplify import OperatorBudget

def do_backpropagation(circuit: QuantumCircuit, observable: SparsePauliOp,
                       op_budget: OperatorBudget, depth:int=1) -> QuantumCircuit:
    
    slices = slice_by_depth(circuit, depth)
    bp_obs, remaining_slices, metadata = backpropagate(
            observable, slices, operator_budget=op_budget, max_seconds=180
        )
    bp_circuit = combine_slices(remaining_slices, include_barriers=True)

    return bp_circuit

def perform_obp(circuit: QuantumCircuit, observable: SparsePauliOp,
               op_budget: OperatorBudget, backend, depth:int=1) -> int:

    if circuit.num_clbits > 0: # circuit has measurement
        circuit.remove_final_measurements(inplace=True)
        if circuit.num_clbits > 0: # implies mid circuit measurements
            return None, None

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=1)

    # first perform backpropagation without transpilation
    bp_circuit_no_transpile = do_backpropagation(circuit, observable, op_budget)
    if type(bp_circuit_no_transpile) != QuantumCircuit:
        depth_transpile_after_bp = None
    else:
        transpiled_bp_circuit = pm.run(bp_circuit_no_transpile)
        depth_transpile_after_bp = transpiled_bp_circuit.depth()

    # next perform backpropagation after transpilation
    transpiled_circuit = pm.run(circuit)
    isa_observable = observable.apply_layout(transpiled_circuit.layout)
    bp_circuit_transpile = do_backpropagation(transpiled_circuit, isa_observable, op_budget)
    if type(bp_circuit_transpile) != QuantumCircuit:
        depth_transpile_before_bp = None
    else:
        depth_transpile_before_bp = bp_circuit_transpile.depth()

    return depth_transpile_after_bp, depth_transpile_before_bp
    
