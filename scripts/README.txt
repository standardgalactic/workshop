Supporting scripts for the essay "Hierarchical Synchronization".

round_length.py
Two-worker simulation of infrequent synchronization. Each round, both workers load the common checkpoint, train H local AdamW steps on their own data shard, and return a dense parameter shift. The root combines the shifts by plain averaging or by Nesterov outer momentum. Compute and communication time are modeled, not measured: a round costs H times the step time plus the time to send a shift up and a checkpoint down at the stated bandwidth and latency. The target validation loss is 1.10 times the final loss of a per-step parameter averaging reference run. Options expose the intervals, budget, links, outer learning rate, outer momentum, seeds and target slack.

diag_experiment.py
Fixed-point diagnostic and heterogeneity runs. For a chosen shard heterogeneity (mild or strong) and local optimizer (AdamW or memoryless SGD), it runs the per-step reference, takes its final iterate as the approximate reference solution, runs plain averaging at each interval on the WAN cost model, and estimates the full-participation mean local shift at the reference from independent data-stream draws. For AdamW it does this both with the reference run's per-worker optimizer state restored and with fresh state.

check_scalar.py
Exact-arithmetic check of the scalar results in the essay, using rational numbers. It confirms the closed-form fixed point and bias formula against the direct computation, iterates the plain round map and the idealized control-variate round numerically for each horizon in the bias table, evaluates the contraction factor of the control-variate map, reproduces the three-worker double-counting example with its inclusion-exclusion correction, and checks the fixed-point diagnostic D_H at the reference solution: the closed form against a direct exact computation, the relation to the fixed-point bias, the tabulated values, nonvanishing for every horizon beyond one step, and exact vanishing under the idealized control-variate round. It has no dependencies beyond the Python standard library.

run_all.sh
The exact commands behind the reported tables, in the order they appear in the essay.

results/
JSON output of the runs reported in the essay. round_length_results.json is the interval table, sweep holds the nine outer-parameter settings at H = 500, and diagnostic holds the four heterogeneity and optimizer configurations together with their logs. Times are modeled seconds and are averaged over two seeds in the essay's tables.

Notes. Python 3 with PyTorch is required, and CPU is sufficient. Seeds fix the teacher, shard distributions, initialization and data streams, but thread scheduling can change low-order digits of the floating-point results.
