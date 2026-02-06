
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)
from qiskit_addon_cutting import cut_wires, expand_observables,partition_problem, generate_cutting_experiments
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.simplify import OperatorBudget



def do_backpropagation(circuit: QuantumCircuit, observable: SparsePauliOp,
                       op_budget: OperatorBudget, depth:int=1):
    
    slices = slice_by_depth(circuit, depth)
    bp_obs, remaining_slices, metadata = backpropagate(
            observable, slices, operator_budget=op_budget, max_seconds=60
        )
    bp_circuit = combine_slices(remaining_slices, include_barriers=False)

    return bp_obs, bp_circuit



def cutting (circuit: QuantumCircuit, observable: SparsePauliOp, backend):

    optimization_settings = OptimizationParameters(seed=111)
    device_constraints = DeviceConstraints(qubits_per_subcircuit = (circuit.num_qubits/2)+1)

    
    cut_circuit, metadata = find_cuts(circuit, optimization_settings, device_constraints)
    num_cuts = len(metadata["cuts"])
    qc_w_ancilla = cut_wires(cut_circuit)
    observables_expanded = expand_observables(observable.paulis, circuit, qc_w_ancilla)


    partitioned_problem = partition_problem(circuit = qc_w_ancilla, observables = observables_expanded)
    subcircuits = partitioned_problem.subcircuits
    subobservables = partitioned_problem.subobservables
    sampling_overhead= np.prod([basis.overhead for basis in partitioned_problem.bases]) #calculates sampling overhead
    if sampling_overhead <= 1e+12:

        subexperiments, coefficients = generate_cutting_experiments(circuits = subcircuits, 
                                                        observables = subobservables, num_samples = np.inf)
        total_subexperiments = sum(len(subexperiments[i]) for i in list(subexperiments.keys()))
        #print(total_subexperiments)
        return num_cuts, total_subexperiments



def perform_cutting_copy(circuit: QuantumCircuit, observable: SparsePauliOp, backend, op_budget: OperatorBudget) -> int:
    # if circuit.num_clbits > 0: # circuit has measurement
    #     circuit.remove_final_measurements(inplace=True)    
    pm = generate_preset_pass_manager(optimization_level=3, basis_gates=backend.configuration().basis_gates, seed_transpiler=1)
    synth_circuit = pm.run(circuit)
    # First perform cutting without OBP
    try:
        number_of_cuts, num_subexps_cut = cutting(synth_circuit, observable, backend)
        
        # Now perform OBP then cutting
        bp_obs, bp_circuit = do_backpropagation(synth_circuit, observable, op_budget)
        #print(type(bp_circuit))
        if type(bp_circuit) != QuantumCircuit:
            return number_of_cuts,None,num_subexps_cut, None
        else:
            number_of_obp_cuts, num_subexps_obp_cut = cutting(bp_circuit, bp_obs, backend)
            #print(num_subexps_obp_cut)
            return number_of_cuts, number_of_obp_cuts,num_subexps_cut, num_subexps_obp_cut
    except:
        return None, None, None,None
    