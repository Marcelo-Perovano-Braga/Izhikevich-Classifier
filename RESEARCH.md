# Robustness study with a frozen checkpoint

`evaluate_research.py` extends the existing research evaluation in three areas:
robustness, held-out conditions, and comparison with simple methods. It runs
outside the graphical interface in the same Python environment and produces
a local HTML report. No additional dependencies are required.

The study loads `weights/classifier.pt` and preserves its weights and
normalization. It does not retrain, recalibrate, or reselect the CNN. Original
training/test results remain in their files; each study gets a new directory.
The command verifies that the checkpoint and training database did not change
during execution. Their configurations and SHA-256 hashes enter the report.

## Running the study

From the repository root, with the project's Python environment activated:

```powershell
python evaluate_research.py
```

Defaults are **100 neurons per class**, seeds **101, 202, and 303**, and the 23
conditions below, using the original training metadata defaults: 69 runs and
20,700 neuron observations. Each condition/seed generates a population;
the report aggregates across seeds. In a network, observations share
connections and are not 300 independent experimental replicates.

You need the trained checkpoint and the **original database** that supplied
its training set. If they are elsewhere:

```powershell
python evaluate_research.py --checkpoint weights/classifier.pt --database data/neural_patterns.sqlite
```

The script uses only SQLite's `train` split to fit the baselines. Configurations
recorded in the database must match those in the checkpoint. The legacy
checkpoint does not contain its source database's hash: matching configurations
does not prove that the operator supplied the exact original database.
Evaluation traces are also checked for exact matches with loaded training
traces. This check does not rule out every possible form of leakage.

If the original database is unavailable, explicitly select this alternative:

```powershell
python evaluate_research.py --baseline-source synthetic --synthetic-fit-per-class 500
```

It generates a new fitting set with constant current, seed 17, and a random
stream separate from evaluation. Examples with insufficient activity are
filtered only in this fitting set. The report identifies `explicit_synthetic_fit`:
the baseline did not see the same training rows as the CNN, so the comparison
is exploratory. The program never substitutes random weights for a missing
checkpoint and never regenerates the original database automatically.

For a smaller smoke test, whose results must not be confused with the full
protocol:

```powershell
python evaluate_research.py --samples-per-class 20 --seeds 101 202 303
```

To increase population size while retaining replicates:

```powershell
python evaluate_research.py --samples-per-class 1000 --seeds 101 202 303 --out-degree 100 --threads 8
```

This example uses 3,000 neurons per run and 300,000 synapses under connected
conditions. Population size changes the experiment; compare studies of the
same size. The study limit is 10,000 neurons per class, and the network retains
a 10-million-synapse budget. This study measures classification, not the
computer's performance limit.

The command prints progress and the path to `report.html`. Open that file in
a browser. Explicitly supplied paths are relative to the current directory;
default paths are relative to the project root.

## 1. Robustness sweeps

| Axis | Protocol values | What stays fixed |
|---|---|---|
| Reference | Constant current, independent neurons | Checkpoint metadata |
| Observation noise | 0, training value, 1, 2, 5, 10 mV | Parameters and current per neuron |
| Parameter variation | 0, training value, 0.06, 0.10, 0.15 | Latent draws, current, and noise |
| Synaptic strength | 0, 0.1, 0.5, 1 | Topology, parameters, current, and noise |
| Pulse onset | 10%, 30%, 50% of the window | Pulse family and neuron draws |

Duplicate values within an axis are removed, so the number of conditions may
change if the checkpoint uses different noise or variation values. The
zero-strength control reproduces reference traces bit for bit.

The pulse lasts 50% of the window. Comparisons between onset times isolate
that shift within the pulse family. Comparing it with the constant-current
reference also changes stimulus shape and total charge.

The network uses distinct random directed connections, no self-connections,
20% inhibitory sources, inhibitory gain 4, a 1 ms delay, and 5 ms decay.
`--out-degree` sets outgoing connections per neuron, capped at `N-1`. Strength
is the excitatory amplitude; inhibitory amplitude receives a negative sign and
the configured gain. Excitatory/inhibitory roles are independent of RS/CH/FS
labels.

Conditions within an axis reuse latent draws for each seed. Changing variation
changes their scale without drawing a new population. Noise is added after
simulation and does not create physical spikes or synaptic events. It follows
the original generator's voltage clipping; with zero noise, clipping is not
applied. This axis therefore evaluates the project's complete observation
channel, not just an unclipped Gaussian perturbation.

## 2. Held-out conditions

The fitting set uses only constant inputs within the recorded range. Evaluation
includes these families/ranges, which it never receives:

| Condition | Definition |
|---|---|
| Pulses | Amplitude sampled from the training range, zero outside the pulse |
| Ramp | Amplitude multiplied by a linear ramp from 0.25 to 1.75 |
| Sinusoid | Amplitude multiplied by `1 + 0.5 sin(2π · 20 Hz · t)` |
| Lower current | From `minimum - half the range width` to a float32 strictly below the minimum |
| Upper current | From a float32 strictly above the maximum to `maximum + half the range width` |

For the default 5–30 range, new ranges are approximately -7.5–5 and 30–42.5,
excluding boundaries 5 and 30. Sampling and float32 conversion preserve that
exclusion. Ramp/sinusoid inputs can also exceed the original domain in
instantaneous amplitude; the CSV flags this per neuron.

“Held out” means excluded from the constant-current training protocol
implemented in this repository, subject to the legacy checkpoint's provenance
limitation. It is not evidence of performance on biological signals.

The Gaussian parameter-variation distribution has unbounded support. Increasing
its spread creates a distribution shift without producing a rigorously unseen
parameter range. The report makes this distinction.

## 3. Temporal-feature baselines

Two nearest-centroid classifiers are fitted on training rows. Each feature is
standardized using **only** its training mean and standard deviation;
near-zero deviations receive a scale of 1. The class with the smallest
Euclidean distance to the standardized centroid is then selected. Ties use the
first class in RS, CH, FS order. There is no test-set hyperparameter search.

| Model | Observed features |
|---|---|
| `rate_centroid` | Spike rate |
| `timing_centroid` | Rate, median ISI, ISI CV, short-ISI fraction, last/first ISI ratio |

An observed spike starts at the edge of a contiguous region with voltage
≥29 mV; a region spanning several samples counts once. Rate uses the duration
`sample count × dt`. Intervals are differences between these spike times.
“Short” means ≤5 ms, fixed before evaluation and recorded by `--burst-isi-ms`.
This is a temporal feature, not a biological burst diagnosis. CV and the
adaptation ratio require three spikes; other ISI features require two.
Features without enough intervals receive zero.

Both baselines receive only features extracted from the **same observed
signals** sent to the CNN. Parameters, expected classes, inputs, and clean
spikes enter none of the predictors. Baseline distances are not calibrated
probabilities.

## Metrics and interpretation

- **Overall accuracy:** correct / all requested neurons. Silent or ambiguous
  examples remain in the denominator. Prediction failures count as errors and
  have their own confusion-matrix column.
- **Macro F1:** the mean F1 across three classes, with zero for a class without
  true positives. A confusion matrix is provided per condition/seed/model.
- **Accuracy among active neurons:** a secondary analysis requiring at least
  the minimum observed spike count recorded in the checkpoint; the same subset
  is applied to all three models.
- **CNN coverage:** the fraction with a finite signal, sufficient observable
  activity, and confidence above the recorded threshold. Accuracy among
  accepted predictions is always accompanied by coverage; 100% on a few
  predictions is not equivalent to 100% overall.
- **NLL, multiclass Brier score, and ECE:** evaluate CNN probabilities without
  fitting a calibrator. Brier sums the three squared errors per example; ECE
  uses ten equal-width confidence bins and does not prove universal calibration.
- **Variation across seeds:** the sample standard deviation of per-run
  accuracies, unavailable with one seed. It is not a confidence interval that
  treats network neurons as independent observations.
- **Inference time:** CNN timing includes copying, normalization, and batches;
  each baseline includes extraction of all five features and its own prediction.
  It excludes population generation, fitting, and metrics. These are diagnostic
  timings without warm-up, not a controlled performance benchmark.

The diagnostic CNN path predicts even for currents outside the original range;
it does not use current information to reject them in advance. This allows
measurement of failures on those examples. The application's rejection policy
remains a different operational policy. Secondary study coverage uses
activity/confidence without this prior current filter.

Evaluation labels identify the **source parameter family** RS/CH/FS. A neuron
can change its response under another stimulus or recurrence: correctly
identifying its origin does not necessarily identify its current firing pattern.

Physical counts use threshold events ≥30 mV **before noise** in both independent
and connected populations. The independent study uses the connected simulator's
counter with degree zero and equivalent traces. These counts are diagnostics,
not classifier inputs. The interface's legacy counter for independent neurons
remains separate; do not treat its sample counts as identical to these events.

If the rate baseline tracks the CNN, separation may depend primarily on
frequency. If temporal features also retain high accuracy, they offer a simple
explanation of separation in this simulation. A CNN advantage under held-out
conditions must be shown with comparable denominators and replication. Low
accuracy alone does not demonstrate greater realism.

## Outputs and reproducibility

Each run creates `artifacts/research/run_<timestamp>/`:

| File | Contents |
|---|---|
| `report.html` | Comparison map, coverage, counts, variation, and held-out conditions; works offline |
| `report.md` | Text version of the report |
| `study.json` | Full protocol, versions, code/data/checkpoint hashes, seeds, normalizations, metrics, and aggregates |
| `summary.csv` | One row per condition/seed/model, including coverage, calibration, and timings |
| `predictions.csv` | Per-neuron parameters and predictions for auditing |
| `status.json` | `running`, `complete`, `failed`, or `interrupted` status |

Full traces are not saved to avoid enormous files; the protocol, code version,
metadata, and seeds allow regeneration in the same environment. Predictions also
depend on the checkpoint identified by its hash. Different kernels or versions
can produce small numerical differences.

The script rejects a nonempty output directory. Interruptions leave a CSV with
a `.partial` suffix and incomplete status; that material is not a completed
study. Outputs are ignored by Git by default, as are the checkpoint and database.

To list or select conditions:

```powershell
python evaluate_research.py --list-conditions
python evaluate_research.py --conditions reference noise_5 network_0.5 ramp --seeds 101 202 303
```

Subset selection is explicit in results; do not present it as evaluation of the
entire protocol. To repeat a study, use the same commit, checkpoint, database
(or explicit synthetic option), arguments, and environment, with another empty
output directory. The default configuration includes a paired reference;
without selecting it, the relative difference is unavailable.
