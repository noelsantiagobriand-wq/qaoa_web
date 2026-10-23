"""Algorithms adapted from the project's qaoa.ipynb.

The signed scoring convention is preserved:
    cut edge   -> -weight
    uncut edge -> +weight
Lower scores are better.
"""
import random

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


def generate_erdos_graph(vertices=10, probability=0.5, seed=67, output=False):
    """Generate an Erdős–Rényi G(n, probability) graph."""
    return nx.erdos_renyi_graph(vertices, probability, seed=seed)


def generate_constant_graph(vertices=10, degree=3, output=False, seed=67):
    """Generate a random regular graph, validating its mathematical constraints."""
    if degree < 0 or degree >= vertices:
        raise ValueError("Degree must be between 0 and vertices - 1.")
    if vertices * degree % 2:
        raise ValueError(
            f"A {degree}-regular graph with {vertices} vertices is impossible: "
            "vertices × degree must be even. Choose another degree or vertex count."
        )
    return nx.random_regular_graph(d=degree, n=vertices, seed=seed)


def generate_cycle_with_chords(vertices=8, output=False, seed=67):
    """Generate a cycle with floor(vertices / 4) randomly selected chords."""
    if vertices < 3:
        raise ValueError("A cycle needs at least 3 vertices.")
    graph = nx.cycle_graph(vertices)
    num_chords = vertices // 4
    candidates = [
        (u, v)
        for u in range(vertices)
        for v in range(u + 1, vertices)
        if not graph.has_edge(u, v)
    ]
    rng = random.Random(seed)
    if num_chords:
        graph.add_edges_from(rng.sample(candidates, num_chords))
    return graph


def calculate_cut_value(G, partition):
    """Project's signed score: lower is better."""
    total = 0.0
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        if partition[u] != partition[v]:
            total -= weight
        else:
            total += weight
    return total


def actual_cut_size(G, partition):
    """Conventional Max-Cut objective: total weight of crossing edges."""
    return sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if partition[u] != partition[v]
    )


def greedy_partition(G):
    """The project's greedy assignment/tie-breaking rule, without plotting."""
    partition = {}
    for vertex in G.nodes:
        a_neighbours = 0
        b_neighbours = 0
        for neighbor in G.neighbors(vertex):
            if neighbor in partition:
                weight = G[vertex][neighbor].get("weight", 1.0)
                if partition[neighbor] == 0:
                    a_neighbours += weight
                elif partition[neighbor] == 1:
                    b_neighbours += weight

        if a_neighbours > b_neighbours:
            partition[vertex] = 1
        elif b_neighbours > a_neighbours:
            partition[vertex] = 0
        else:
            count_a = sum(1 for v in partition if partition[v] == 0)
            count_b = sum(1 for v in partition if partition[v] == 1)
            partition[vertex] = 0 if count_a <= count_b else 1
    return partition


def greedy_max_cut(G):
    """Compatibility wrapper returning the project's original-style outputs."""
    partition = greedy_partition(G)
    subset_a = [node for node, part in partition.items() if part == 0]
    subset_b = [node for node, part in partition.items() if part == 1]
    return calculate_cut_value(G, partition), subset_a, subset_b


def exact_max_cut(G):
    """Exact solver by exhaustive enumeration (fixes the first node to group 0)."""
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return 0, {}
    best_cut = -float("inf")
    best_partition = None

    # A partition and its complement have the same cut; fix node 0 in group 0.
    for mask in range(1 << (n - 1)):
        partition = {nodes[0]: 0}
        for i in range(1, n):
            partition[nodes[i]] = (mask >> (i - 1)) & 1
        cut = sum(
            G[u][v].get("weight", 1.0)
            for u, v in G.edges()
            if partition[u] != partition[v]
        )
        if cut > best_cut:
            best_cut = cut
            best_partition = partition
    return best_cut, best_partition


def solve_max_cut_qaoa(G, p=1, shots=2048, seed=0, max_iterations=100):
    """QAOA solver from the project notebook; minimizes the signed score."""
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    edges = [
        (index[u], index[v], G[u][v].get("weight", 1.0))
        for u, v in G.edges()
    ]

    rng = np.random.default_rng(seed)
    backend = AerSimulator()
    evaluations = 0

    def run_circuit(params):
        gammas = params[:p]
        betas = params[p:]
        qc = QuantumCircuit(len(nodes), len(nodes))
        qc.h(range(len(nodes)))
        for gamma, beta in zip(gammas, betas):
            for u, v, weight in edges:
                qc.rzz(2 * gamma * weight, u, v)
            for q in range(len(nodes)):
                qc.rx(2 * beta, q)
        qc.measure(range(len(nodes)), range(len(nodes)))
        simulator_seed = int(rng.integers(0, 2**32 - 1))
        return backend.run(
            qc, shots=shots, seed_simulator=simulator_seed
        ).result().get_counts()

    def signed_score(bits):
        ordered = bits[::-1]
        return sum(
            -weight if ordered[u] != ordered[v] else weight
            for u, v, weight in edges
        )

    def objective(params):
        nonlocal evaluations
        evaluations += 1
        counts = run_circuit(params)
        total_shots = sum(counts.values())
        return sum(
            signed_score(bits) * count for bits, count in counts.items()
        ) / total_shots

    initial = rng.uniform(0, 2 * np.pi, 2 * p)
    result = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": max_iterations},
    )

    counts = run_circuit(result.x)
    total_shots = sum(counts.values())
    best_bits = min(counts, key=signed_score)
    best_ordered = best_bits[::-1]
    partition = {
        nodes[i]: int(best_ordered[i]) for i in range(len(nodes))
    }
    best_score = calculate_cut_value(G, partition)
    expected_score = sum(
        signed_score(bits) * count for bits, count in counts.items()
    ) / total_shots

    return {
        "partition": partition,
        "best_score": best_score,
        "expected_score": expected_score,
        "evaluations": evaluations,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
