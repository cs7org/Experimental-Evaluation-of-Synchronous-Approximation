# Historical solver-tolerance experiment

This folder contains the located source for the separate arithmetic-accuracy
experiment associated with the `p_full_vs_digits` and `runtime_vs_digits`
figures (Figures 22 and 23 in the manuscript filenames).

## Outstanding issues before publication

This bundle is preserved as found; it is not yet a verified reproduction of
the final Figures 22 and 23.

- `methods/dual_bounds_method.py` still excludes auxiliary phase `j=-1` from
  the expanded slower customer-count aggregation. The correction in the main
  package's `methods/fast_slow.py` does not apply to this separate module.
- The runner includes Erlang 10, CME 10, Faster 10, and Slower 10, but does not
  include the Erlang 10/D curve present in the current Figure 22.

These issues must be reconciled before publishing this bundle as final
figure-reproduction code. No historical results are included here.

## Contents

- `arithmetic_accuracy.py`: experiment runner.
- `methods/`: nine Python modules used by this runner.
- `methods/iltcme.json`: the CME coefficient table.
- `requirements.txt`: the dependency list from the original experiment.

This folder uses a different `methods` namespace from the main package.
Run it from this folder and do not merge the two `methods` directories.

## Setup and running

Use Python 3.12 or a compatible recent Python version. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\arithmetic_accuracy.py --help
.\.venv\Scripts\python.exe .\arithmetic_accuracy.py
```

The defaults use capacity 10, arrival rate 0.1, mean service time 1, and
residual precision parameters 5 through 30. The parameter is a solver
tolerance index; it is not necessarily the number of arithmetic digits used.
Outputs are written to `results/arithmetic_accuracy/` and include SVG plots,
sample/failure CSV files, and a JSON result file.
