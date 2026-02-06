from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)
from qiskit_addon_cutting import cut_wires
from qiskit_addon_cutting import partition_problem

def do_cc(circuit: QuantumCircuit, backend) -> QuantumCircuit:
    if circuit.num_clbits > 0: # circuit has measurement
        circuit.remove_final_measurements(inplace=True)
        if circuit.num_clbits > 0: # implies mid circuit measurements
            return None, None, None
        
    optimization_settings = OptimizationParameters(seed=111)
    device_constraints = DeviceConstraints(qubits_per_subcircuit = circuit.num_qubits//2)
    
    pm = generate_preset_pass_manager(optimization_level=3, basis_gates=backend.configuration().basis_gates, seed_transpiler=1)
    # if type(circuit) != QuantumCircuit:
    #     number_of_cuts = None
    #     sampling_overhead_values = None
    #     subcirc_length = None
    # else:
    synth_circuit = pm.run(circuit)
    
    # Specify the size of the QPUs available
    cut_circuit, metadata = find_cuts(synth_circuit, optimization_settings, device_constraints)
        
    number_of_cuts = len(metadata["cuts"])
    sampling_overhead_values = metadata["sampling_overhead"]
    subcirc_length={}
    if number_of_cuts == 0:
        subcirc_length[0] = circuit.num_qubits
    else:
        qc_w_ancilla = cut_wires(cut_circuit)
        partitioned_problem = partition_problem(circuit=qc_w_ancilla)

        subcircuits = partitioned_problem.subcircuits
        for key, value in subcircuits.items():
            subcirc_length[key] = value.num_qubits

    return number_of_cuts, sampling_overhead_values, subcirc_length

