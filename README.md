# QAOA Max-Cut Lab (Streamlit)

Interactive website for comparing the project's greedy heuristic, QAOA, and exact classical solver.

## Included defaults

- Graph family: Erdős–Rényi
- Vertices: 10
- Independent graph instances: 17
- QAOA restarts per graph: 3
- QAOA depth: `p = 3`
- Shots: 512
- Maximum optimizer iterations per restart: 10
- Erdős–Rényi edge probability: 0.5

## Scoring convention

This app preserves the project's scoring function:

- crossing edge: subtract its weight
- uncut edge: add its weight
- lower signed score is better

The conventional cut size (number/weight of crossing edges) is also displayed for clarity. `gap_from_optimum = algorithm_signed_score - exact_signed_score`, so a gap of zero means an optimal solution.

## Run locally

Use Python 3.10+ (Python 3.11 is a sensible choice for a clean environment).

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Deploy

Push `app.py`, `algorithms.py`, and `requirements.txt` to a GitHub repository. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.

## Notes

- QAOA is executed using Qiskit Aer on a classical simulator; these measurements do not demonstrate quantum hardware speedup.
- QAOA's reported best score is the best observed bitstring from the final measurement distribution of the best restart. The mean restart score is shown separately.
- The exact solver exhaustively enumerates equivalent cut partitions, fixing the first node to group 0. At 20 vertices this requires `2^19` partitions per graph; large repeated runs can take time.
- Random regular graphs require `vertices * degree` to be even.
- Each graph instance uses a reproducible seed derived from the base seed.
