
import math
import random
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit_addon_cutting.automated_cut_finding import (
    find_cuts,
    OptimizationParameters,
    DeviceConstraints,
)
from qiskit_addon_cutting import cut_wires, expand_observables,partition_problem, generate_cutting_experiments
import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_utils.slicing import slice_by_depth, combine_slices
from qiskit_addon_obp import backpropagate
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_ibm_runtime.fake_provider import FakeTorino
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_aer import AerSimulator
def cost_function(params, ansatz, hamiltonian, estimator):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    return energy
def objective_function(w : int) -> int:
    #circuit = QuantumCircuit.from_qasm_file("/Users/debarthipal/Library/CloudStorage/OneDrive-IBM/Desktop/VS files/cutting+obp_new_git_env/QASM_circuits_large/knn_n67.qasm")
    circuit=efficient_su2(6, entanglement='linear', reps=3)
    np.random.seed(42)
    x0 = 2 * np.pi * np.random.random(circuit.num_parameters)
    
    
    
    #circuit.remove_final_measurements(inplace=True)
    n=circuit.num_qubits
    observable_terms = [
            "I"*(j-1) + "Z" + "I"*(n - j)
            for j in range(1, n+1)
        ]
    observable = SparsePauliOp(observable_terms, coeffs=[1/(n)] * (n))
    backend =FakeTorino()
    with Session(backend=AerSimulator()) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 10000

        res = minimize(
            cost_function,
            x0,
            args=(circuit.copy(), observable, estimator),
            method="cobyla",
            options={'maxiter':4000}
        )
    circuit=circuit.assign_parameters(res.x)
    pm = generate_preset_pass_manager(basis_gates=backend.configuration().basis_gates, optimization_level=3, seed_transpiler=1)
    synth_circuit = pm.run(circuit)

    optimization_settings = OptimizationParameters(seed=111)
    device_constraints = DeviceConstraints(qubits_per_subcircuit = (synth_circuit.num_qubits/2)+1)

    slices = slice_by_depth(synth_circuit, max_slice_depth=1)
    op_budget =OperatorBudget(max_qwc_groups = w)
    #print(op_budget)
    bp_obs, remaining_slices, metadata = backpropagate(
            observable, slices, operator_budget=op_budget, max_seconds=60
        )
    bp_circuit = combine_slices(remaining_slices, include_barriers=False)
    #print(type(bp_circuit))
    if type(bp_circuit) !=QuantumCircuit:
        return 0
    else:

        cut_circuit, metadata = find_cuts(bp_circuit, optimization_settings, device_constraints)
        qc_w_ancilla = cut_wires(cut_circuit)
        observables_expanded = expand_observables(bp_obs.paulis, bp_circuit, qc_w_ancilla)


        partitioned_problem = partition_problem(circuit = qc_w_ancilla, observables = observables_expanded)
        subcircuits = partitioned_problem.subcircuits
        subobservables = partitioned_problem.subobservables

        subexperiments, coefficients = generate_cutting_experiments(circuits = subcircuits, 
                                                        observables = subobservables, num_samples = np.inf)
        total_subexperiments = sum(len(subexperiments[i]) for i in list(subexperiments.keys()))
        print(f"total subexperiments for max_qwc_group {w} is",total_subexperiments)
        return total_subexperiments

# Neighbor function: small random change
def get_neighbour(w : int, step_size:int=1) -> int:
    lower_bound = w - step_size if w - step_size > 0 else 1
    upper_bound = w + step_size if w + step_size < 41 else 40
    neighbor = random.randint(lower_bound, upper_bound)
    return neighbor 

# Simulated Annealing function
def perform_simulated_annealing(bounds, n_iterations, step_size, temp):
    # Initial solution
    best = random.randint(bounds[0][0], bounds[0][1])
    best_eval = objective_function(best)
    #print(f"the best w is", best)
    #current, current_eval = best, best_eval
    scores = [(best,best_eval)] # minima so far
    subexperiment_counts = {best: best_eval}  # initial point

    for i in range(n_iterations):

        # Decrease temperature
        t = temp / float(i + 1) 

        # Generate candidate solution
        candidate = get_neighbour(best, step_size)
        print(f"Value of w for iteration {i} is :", candidate)
        if candidate in subexperiment_counts:
            candidate_eval = subexperiment_counts[candidate]
        else:
            candidate_eval = objective_function(candidate)
            subexperiment_counts[candidate] = candidate_eval

        # Check if this is the optimal minima so far
        print(f"best_eval is :", best_eval)
        print(f"candidate_eval is :", candidate_eval)
        #print(f"current_eval is :", current_eval)
        if candidate_eval < best_eval:
            #current, current_eval = candidate, candidate_eval
            best, best_eval = candidate, candidate_eval
            scores.append((best,best_eval))
        
        # if candidate_eval < current_eval:
        #     current, current_eval = candidate, candidate_eval
        else:
            delta_e = candidate_eval - best_eval
            prob = math.exp(- delta_e / t)
            print(f"prob is", prob)
            create_val = np.random.rand(1)[0]
            if create_val <= prob:
                best, best_eval = candidate, candidate_eval

        # Optional: print progress
        # if i % 10 == 0:
        #     print(f"Iteration {i}, Temperature {t:.3f}, Best Evaluation {current_eval:.5f}")

    return best, best_eval, scores

