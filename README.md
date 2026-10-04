# Workshop

Short, self-contained essays that correct common misreadings of results in recent papers and reports. Each essay is anchored on one source, states what the evidence does and does not establish, and backs its claims with a small simulation or worked example that can be rerun from this repository. The simulations are illustrations of mechanisms, labeled as such in the text, not reproductions of the original studies.

## Reading order

The order below moves from how averages and attributions mislead, through what tests and metrics can show, to what training signals and derivations certify. Each essay stands alone, so any entry point works.

### Averages and attribution

1. [What Averaging Discards: Multi-Anchor Exploration and the Limits of Its Evidence](https://standardgalactic.github.io/workshop/multi_anchor_essay.pdf)
   A multi-anchor exploration paper and the claim that single-goal prediction is intrinsically mismatched to exploration. An audio overview is included as `Why_AI_Robots_Crash_Into_Walls`.
2. [Where the Averaging Goes: Endpoint Regression, Local Velocity Fields, and the Last Euler Step](https://standardgalactic.github.io/workshop/averaging.pdf)
   Why "regression averages, generation does not" is half right, and where the averaging goes in a flow-matching model.
3. [Increments Are Not Contributions: Sequential Ablation, Interaction, and What a Table Can Establish](https://standardgalactic.github.io/workshop/increments.pdf)
   Why the difference between consecutive ablation rows is not the intrinsic contribution of a component.
4. [Same Backbone, Different Everything: Attributing a Benchmark Gain to an Interface, a Search Budget, and a Test-Time Portfolio](https://standardgalactic.github.io/workshop/backbone.pdf)
   A comparison that holds the model fixed while varying everything else, and how to split the gain among the pieces.

### What tests and metrics can show

5. [What the Metric Can See: Positive Controls, Dynamic Range, and the Interpretation of Null Benchmarks](https://standardgalactic.github.io/workshop/metric.pdf)
   When a sophisticated model ties a trivial predictor, whether the model or the metric is the limit.
6. [Overlapping Error Bars Prove Nothing: Paired Inference and the Unit of Replication](https://standardgalactic.github.io/workshop/errorbars.pdf)
   What overlapping intervals and very small P values do and do not say about two methods.
7. [Which Trials Count: Inclusion Rules, Influence, and the Reproducibility of a Pooled Estimate](https://standardgalactic.github.io/workshop/trials.pdf)
   How a pooled meta-analytic estimate depends on the set of trials that enter it.
8. [What a Correlation of 0.18 Carries: Pooled Cohorts and the Step From a Culture Experiment to a Population Pattern](https://standardgalactic.github.io/workshop/pooledcorr.pdf)
   What a pooled-cohort rank correlation can and cannot say about a mechanism shown in culture.
9. [Where Covariance Comes From: Log-Scale Slopes, Raw-Scale Slopes, and the Attribution of Correlation to the Asymptomatic](https://standardgalactic.github.io/workshop/covariance.pdf)
   Why a steeper log-link slope at low counts does not by itself locate where covariation comes from.

### Training signals and derivations

10. [Privileged Teachers: What an Oracle Label Teaches a Selector Under Partial Observation](https://standardgalactic.github.io/workshop/teachers.pdf)
    Why labels that are correct for an oracle need not be the best labels for an agent that sees less.
11. [A Proof of the Wrong Thing: What a Verified Derivation Certifies and What It Leaves Open](https://standardgalactic.github.io/workshop/proofwrong.pdf)
    What a checked derivation in wireless communications establishes, and the links a checker does not supply.
12. [Counting Edges and Counting Equations: What the Dimension-Matching Result Establishes, and What the Network Experiments Can and Cannot Confirm](https://standardgalactic.github.io/workshop/modulatability.pdf)
    Dimension counting, rank and conditioning in an oscillator-modulation paper, and how the experimental protocol shapes the pattern.

## Reproducing the figures and numbers

Every script is plain Python 3 and needs `numpy`, `scipy`, `matplotlib` and `statsmodels`.

```
pip install numpy scipy matplotlib statsmodels
python run_all.py            # runs every script
python run_all.py modul nma  # runs only the named scripts
```

Figures and `*_results.json` files are written to `outputs/<script name>/`.

| Script | Essay |
|---|---|
| `sim.py`, `seeds.py` | Where the Averaging Goes |
| `shap.py` | Increments Are Not Contributions |
| `attrib.py` | Same Backbone, Different Everything |
| `calib.py` | What the Metric Can See |
| `paired.py` | Overlapping Error Bars Prove Nothing |
| `nma.py` | Which Trials Count |
| `pooled.py` | What a Correlation of 0.18 Carries |
| `nomorb.py` | Where Covariance Comes From |
| `teach.py` | Privileged Teachers |
| `wireless.py` | A Proof of the Wrong Thing |
| `modul.py` | Counting Edges and Counting Equations |

## Rebuilding the essays

Each essay is one `.tex` file that reads the `fig_*.pdf` files in the same folder.

```
latexmk -pdf -interaction=nonstopmode modulatability.tex
```
