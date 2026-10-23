import io
import time
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st

from algorithms import (
    actual_cut_size,
    calculate_cut_value,
    exact_max_cut,
    generate_constant_graph,
    generate_cycle_with_chords,
    generate_erdos_graph,
    greedy_partition,
    solve_max_cut_qaoa,
)

st.set_page_config(
    page_title="QAOA Max-Cut Lab",
    layout="wide",
)

st.title("QAOA Max-Cut Lab")
st.markdown(
    "Compare a greedy heuristic, QAOA, and an exact classical solver "
    "on the **same graph instances**."
)

with st.sidebar:
    st.header("Experiment settings")
    graph_type = st.selectbox(
        "Graph type",
        ["Erdős–Rényi", "Constant-degree", "Cycle with chords"],
        index=0,
    )
    vertices = st.slider("Number of vertices", min_value=4, max_value=20, value=10)
    graph_instances = st.number_input(
        "Independent graph instances", min_value=1, max_value=100, value=17, step=1
    )
    restarts = st.number_input(
        "QAOA restarts per graph", min_value=1, max_value=10, value=3, step=1
    )
    p = st.slider("QAOA depth (p)", min_value=1, max_value=5, value=3)
    shots = st.selectbox("Shots per circuit evaluation", [256, 512, 1024, 2048], index=1)
    max_iterations = st.slider(
        "Maximum optimizer iterations per restart", min_value=5, max_value=100, value=10, step=5
    )
    base_seed = st.number_input("Base random seed", min_value=0, max_value=2_000_000_000, value=1)
    probability = 0.5
    degree = 3
    if graph_type == "Erdős–Rényi":
        probability = st.slider("Edge probability", 0.05, 1.0, 0.5, 0.05)
    elif graph_type == "Constant-degree":
        degree = st.slider("Degree of each vertex", 1, vertices - 1, min(3, vertices - 1))
        if vertices * degree % 2:
            st.warning(
                "A regular graph requires vertices × degree to be even. "
                "Change the degree or vertex count before running."
            )

    run_button = st.button("Run comparison", type="primary", use_container_width=True)

def make_graph(seed):
    if graph_type == "Erdős–Rényi":
        return generate_erdos_graph(vertices, probability, int(seed))
    if graph_type == "Constant-degree":
        return generate_constant_graph(vertices, degree, seed=int(seed))
    return generate_cycle_with_chords(vertices, seed=int(seed))


def draw_graph(graph, partition=None, title=None):
    pos = nx.spring_layout(graph, seed=42)
    if partition is None:
        node_colors = ["#9bbfe0"] * graph.number_of_nodes()
    else:
        node_colors = [
            "#4c78a8" if partition[node] == 0 else "#f28e2b"
            for node in graph.nodes()
        ]
    fig, ax = plt.subplots(figsize=(10, 5))
    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        edge_color="#9aa0a6",
        node_size=600,
        font_weight="bold",
        font_color="white",
        width=1.4,
    )
    ax.set_title(title or "Generated graph")
    ax.axis("off")
    fig.tight_layout()
    return fig


def partition_text(partition):
    group_a = [str(node) for node, part in partition.items() if part == 0]
    group_b = [str(node) for node, part in partition.items() if part == 1]
    return f"A: {{{', '.join(group_a)}}}  |  B: {{{', '.join(group_b)}}}"


if "experiment_results" not in st.session_state:
    st.session_state.experiment_results = None
if "sample_graph" not in st.session_state:
    st.session_state.sample_graph = None
if "sample_partitions" not in st.session_state:
    st.session_state.sample_partitions = None

if run_button:
    if graph_type == "Constant-degree" and vertices * degree % 2:
        st.error("Cannot generate a regular graph with this vertex/degree combination.")
        st.stop()

    results = []
    run_details = []
    progress = st.progress(0, text="Preparing experiment...")
    status = st.empty()
    total_graphs = int(graph_instances)
    overall_start = time.perf_counter()

    for graph_idx in range(total_graphs):
        graph_seed = int(base_seed) + graph_idx
        status.write(f"Graph {graph_idx + 1}/{total_graphs}: generating graph and solving exact/greedy baselines…")
        graph = make_graph(graph_seed)

        exact_start = time.perf_counter()
        optimum_cut, optimum_partition = exact_max_cut(graph)
        exact_runtime = time.perf_counter() - exact_start
        optimum_score = calculate_cut_value(graph, optimum_partition)

        greedy_start = time.perf_counter()
        greedy_part = greedy_partition(graph)
        greedy_runtime = time.perf_counter() - greedy_start
        greedy_score = calculate_cut_value(graph, greedy_part)
        greedy_cut = actual_cut_size(graph, greedy_part)

        restart_results = []
        qaoa_times = []
        for restart_idx in range(int(restarts)):
            status.write(
                f"Graph {graph_idx + 1}/{total_graphs} · "
                f"QAOA restart {restart_idx + 1}/{int(restarts)} "
                f"(p={p}, shots={shots})"
            )
            qstart = time.perf_counter()
            q = solve_max_cut_qaoa(
                graph,
                p=int(p),
                shots=int(shots),
                seed=graph_seed * 1000 + restart_idx,
                max_iterations=int(max_iterations),
            )
            qruntime = time.perf_counter() - qstart
            qscore = calculate_cut_value(graph, q["partition"])
            qcut = actual_cut_size(graph, q["partition"])
            restart_results.append({
                "restart": restart_idx + 1,
                "qaoa_score": qscore,
                "qaoa_cut": qcut,
                "expected_score": q["expected_score"],
                "runtime_seconds": qruntime,
                "evaluations": q["evaluations"],
                "optimizer_success": q["optimizer_success"],
                "partition": q["partition"],
            })
            qaoa_times.append(qruntime)
            run_details.append({
                "graph_type": graph_type,
                "graph_seed": graph_seed,
                "vertices": vertices,
                "edges": graph.number_of_edges(),
                "restart": restart_idx + 1,
                "p": p,
                "shots": shots,
                "max_iterations": max_iterations,
                "qaoa_score": qscore,
                "qaoa_cut_size": qcut,
                "expected_score": q["expected_score"],
                "gap_from_optimum": qscore - optimum_score,
                "runtime_seconds": qruntime,
                "evaluations": q["evaluations"],
                "optimizer_success": q["optimizer_success"],
            })

        best_restart = min(restart_results, key=lambda item: item["qaoa_score"])
        mean_restart_score = sum(r["qaoa_score"] for r in restart_results) / len(restart_results)
        mean_expected_score = sum(r["expected_score"] for r in restart_results) / len(restart_results)
        best_qaoa_score = best_restart["qaoa_score"]
        best_qaoa_cut = best_restart["qaoa_cut"]

        results.append({
            "graph_type": graph_type,
            "graph_seed": graph_seed,
            "vertices": vertices,
            "edges": graph.number_of_edges(),
            "optimum_score": optimum_score,
            "optimum_cut_size": optimum_cut,
            "greedy_score": greedy_score,
            "greedy_cut_size": greedy_cut,
            "greedy_gap": greedy_score - optimum_score,
            "greedy_optimal": abs(greedy_score - optimum_score) < 1e-9,
            "greedy_runtime_seconds": greedy_runtime,
            "qaoa_best_score": best_qaoa_score,
            "qaoa_best_cut_size": best_qaoa_cut,
            "qaoa_mean_restart_score": mean_restart_score,
            "qaoa_mean_expected_score": mean_expected_score,
            "qaoa_gap": best_qaoa_score - optimum_score,
            "qaoa_optimal": abs(best_qaoa_score - optimum_score) < 1e-9,
            "qaoa_mean_runtime_seconds": sum(qaoa_times) / len(qaoa_times),
            "exact_runtime_seconds": exact_runtime,
            "qaoa_restarts": int(restarts),
            "p": int(p),
            "shots": int(shots),
            "max_iterations": int(max_iterations),
        })

        if graph_idx == 0:
            st.session_state.sample_graph = graph
            st.session_state.sample_partitions = {
                "Exact classical": optimum_partition,
                "Greedy": greedy_part,
                "QAOA (best restart)": best_restart["partition"],
            }

        progress.progress(
            (graph_idx + 1) / total_graphs,
            text=f"Completed {graph_idx + 1} of {total_graphs} graph instances",
        )

    elapsed = time.perf_counter() - overall_start
    st.session_state.experiment_results = pd.DataFrame(results)
    st.session_state.run_details = pd.DataFrame(run_details)
    st.session_state.experiment_elapsed = elapsed
    status.success(f"Experiment completed in {elapsed:.1f} seconds.")
    progress.empty()

df = st.session_state.experiment_results

if df is None:
    st.subheader("Ready to run")
    st.write(
        f"Default configuration: **{graph_type}**, **{vertices} vertices**, "
        f"**{graph_instances} graph instances**, **{restarts} QAOA restarts per graph**, "
        f"**p={p}**, **{shots} shots**."
    )
else:
    st.subheader("Experiment summary")
    total = len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graphs tested", total)
    qaoa_percentages = [
        100 * cut / optimum if optimum > 0 else 100
        for cut, optimum in zip(df["qaoa_best_cut_size"], df["optimum_cut_size"])
    ]
    greedy_percentages = [
        100 * cut / optimum if optimum > 0 else 100
        for cut, optimum in zip(df["greedy_best_cut_size"], df["optimum_cut_size"])
    ]    
    c2.metric("QAOA Mean percentage of optimum", f"{sum(qaoa_percentages) / len(qaoa_percentages):.1f}%")    
    c3.metric("greedy Mean percentage of optimum", f"{sum(greedy_percentages) / len(greedy_percentages):.1f}%")    
    c4.metric("Total runtime", f"{st.session_state.experiment_elapsed:.1f} s")
    
    st.subheader("Partitions — first graph")
    if st.session_state.sample_graph is not None:
        graph = st.session_state.sample_graph
        tabs = st.tabs(["Graph", "Exact classical", "Greedy", "QAOA best"])
        with tabs[0]:
            st.pyplot(draw_graph(graph), clear_figure=True)
            st.write(f"Edges: {list(graph.edges())}")
        sample_map = st.session_state.sample_partitions
        for tab, name in zip(tabs[1:], ["Exact classical", "Greedy", "QAOA (best restart)"]):
            with tab:
                part = sample_map[name]
                score = calculate_cut_value(graph, part)
                cut_size = actual_cut_size(graph, part)
                st.pyplot(draw_graph(graph, part, f"{name} · signed score {score:g}"), clear_figure=True)
                st.write(f"Signed score: **{score:g}** · Cut size: **{cut_size:g}**")
                st.write(partition_text(part))
                
    if "show_other_graphs" not in st.session_state:
        st.session_state.show_other_graphs = False
    
    if st.button(
        "Hide other graphs" if st.session_state.show_other_graphs
        else "Show other graphs",
        use_container_width=True,
    ):
        st.session_state.show_other_graphs = not st.session_state.show_other_graphs
    if st.session_state.show_other_graphs:
        st.markdown("### Average signed score by algorithm")
        score_summary = pd.DataFrame({
            "Algorithm": ["Exact classical", "Greedy", "QAOA (best of restarts)", "QAOA (mean restart)"],
            "Mean signed score": [
                df["optimum_score"].mean(),
                df["greedy_score"].mean(),
                df["qaoa_best_score"].mean(),
                df["qaoa_mean_restart_score"].mean(),
            ],
        }).set_index("Algorithm")
        st.bar_chart(score_summary)
    
        st.markdown("### Per-graph results")
        visible_cols = [
            "graph_seed", "edges", "optimum_score", "greedy_score", "greedy_gap",
            "qaoa_best_score", "qaoa_gap", "qaoa_mean_restart_score",
            "greedy_runtime_seconds", "qaoa_mean_runtime_seconds", "exact_runtime_seconds",
        ]
        st.dataframe(df[visible_cols], use_container_width=True, hide_index=True)
    
        st.markdown("### Optimum gaps")
        gap_df = df[["graph_seed", "greedy_gap", "qaoa_gap"]].set_index("graph_seed")
        st.line_chart(gap_df)
    
        st.markdown("### Download results")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download graph-level results CSV",
            data=csv_bytes,
            file_name="qaoa_maxcut_results.csv",
            mime="text/csv",
        )
        if st.session_state.get("run_details") is not None:
            run_csv = st.session_state.run_details.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download every QAOA restart CSV",
                data=run_csv,
                file_name="qaoa_maxcut_restart_results.csv",
                mime="text/csv",
            )
    
st.divider()
st.caption(
    "Methods: the greedy heuristic follows the project's node-order assignment and tie-break rule; "
    "the exact solver enumerates all unique cut partitions; QAOA uses Qiskit Aer, COBYLA, "
    "RZZ cost layers, RX mixer layers, and the project's signed-score minimization convention."
)
