# Izhikevich Neural Pattern Classifier

A computational proof of concept developed during a university research program
to classify three simulated single-neuron activity patterns: **Regular Spiking
(RS)**, **Chattering (CH)**, and **Fast Spiking (FS)**.

The project uses no biological samples and is not a brain–computer interface.
Each dataset row represents an independent digital neuron instance. The desktop
interface also provides an exploratory sparse synaptic network mode, separate
from the original training setup. Documentation, application text, generated
summaries, and Windows packaging use **US English (en-US)**.

## Project features

- A persistent SQLite dataset with signals, labels, parameters, and metadata.
- Independent, stratified training, validation, and test splits.
- Small parameter variations and observation noise to prevent memorization of
  three deterministic traces.
- Mini-batch training, normalization computed only on training data, early
  stopping, and selection of the best checkpoint using validation data.
- Accuracy, precision, recall, F1, and a confusion matrix on the test set.
- A reproducible comparison of LIF and Izhikevich computational costs.
- An interface showing expected class, predicted class, confidence, and an
  **indeterminate** state for silent, out-of-domain, or low-confidence signals.
- Removal of the old “V(t+1) predictor,” whose target was already in its input.

## Requirements

- Python 3.10–3.14; Python 3.11 remains the recommended option.
- Windows, Linux, or macOS.
- A CUDA GPU is optional. The pipeline uses the CPU when CUDA is unavailable.

The interface uses **PyQt6-Fluent-Widgets** on top of PyQt6. The package provides
cards, controls, status indicators, a confidence bar, and theme switching. Plots
are still rendered by `pyqtgraph`.

### Windows installation

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux/macOS installation

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Training and validation

From the project root:

```bash
python train_pipeline.py --regenerate
```

The command creates:

- `data/neural_patterns.sqlite`: the reproducible database;
- `weights/classifier.pt`: the checkpoint selected using validation data;
- `artifacts/test_metrics.json`: held-out test metrics;
- `artifacts/confusion_matrix.csv`: the confusion matrix;
- `artifacts/training_history.csv`: training history;
- `artifacts/model_benchmark.json`: LIF versus Izhikevich computational costs;
- `artifacts/results_summary.md`: a traceable summary for the final report.

The generated database can be checked without PyTorch:

```bash
python validate_dataset.py
```

The result meets the target only when `target_met` is `true` in
`test_metrics.json`. A working interface does not replace that evidence.

For a smaller infrastructure smoke test:

```bash
python train_pipeline.py --regenerate --samples-per-class 60 --epochs 3 --patience 2
```

Do not use this smoke test as a scientific result.

## Extended evaluation: robustness and held-out conditions

To evaluate an existing checkpoint without retraining, in the same Python
environment:

```powershell
python evaluate_research.py
```

The study compares the frozen CNN with firing-rate and temporal-feature
baselines, sweeps noise, parameter variation, stimulus onset, and synaptic
strength, and evaluates input waveforms and current ranges absent from the
original protocol. Only the original database's training split is used to fit
the baselines. All evaluation neurons, including silent ones, contribute to
overall accuracy; coverage and accuracy among accepted predictions are reported
separately.

The command creates HTML, CSV, and JSON reports in a new directory under
`artifacts/research/`. This tool runs through Python outside the interface; it
does not require rebuilding the executable. See [RESEARCH.md](RESEARCH.md) for
requirements, the protocol, command examples, and interpretation of results.

### Training and adopting a candidate model

To train with more varied inputs and networks, compare against the current
checkpoint, and adopt the candidate only if every criterion is met:

```powershell
python train_robust.py --seed 17029 --adopt-if-superior
```

The expanded protocol, with random envelopes and a fresh confirmation set, is
in [EXPANDED_ROBUST_TRAINING.md](EXPANDED_ROBUST_TRAINING.md). Use `--protocol 2`
to train it; `--protocol 1` remains available to reproduce the first study.

Fitting uses new data with separate splits for training, epoch selection, and
confidence calibration. The comparison uses new seeds, two stimulus families
absent from fitting, and criteria requiring improvements without large
regressions under the original conditions. Successful adoption preserves a
verified copy of the previous checkpoint. See [ROBUST_TRAINING.md](ROBUST_TRAINING.md)
for the criteria, limitations, and instructions for comparing a trained candidate.

## Interface

After generating the checkpoint:

```bash
python main_app.py
```

### Fullscreen and neural background

- **F11** or the expand button in the upper-right corner toggles fullscreen.
- **Esc** exits fullscreen. The previous size is restored; a previously
  maximized window returns to its maximized state.
- Start directly in fullscreen with `python main_app.py --fullscreen`.
  The executable also accepts `--fullscreen` after it has been rebuilt.
- The image button next to the theme control toggles the neural background.
  The background adapts to dark and light themes.

The illustration is static and decorative, unrelated to simulation spikes.
It is bundled with the application and requires no internet connection. The
image keeps its aspect ratio and is cropped centrally to fill the window. The
adjusted version is cached and reused; there is no animation or additional
timer. Resizing, display scaling, theme changes, and toggling the background
update the cache. Charts and tables retain their own surfaces for readability.
If the image file is missing, the interface uses a solid-color background.

The file is `assets/neural_background.png`; its origin, prompt, and usage notes
are in [assets/neural_background.md](assets/neural_background.md).

### Updating an installation with a trained checkpoint

The previous checkpoint remains compatible; retraining is unnecessary.
Install the updated dependencies and run the interface:

```powershell
python -m pip install -r requirements.txt
python main_app.py
```

When extracting the project into a new directory, copy `weights/classifier.pt`
from the previous directory. You can also copy `artifacts/` to retain existing
metrics. The active `.pt` checkpoint and SQLite database are not tracked in this
repository; keep local copies or generate them using the training pipeline.

### Independent populations

Set the **RS**, **CH**, and **FS** neuron counts separately, from zero to
1,000,000 per class, subject to a **1,000,000 total** limit across all three
classes and at least one neuron overall.
The initial population has ten of each pattern. The stimulus control applies
the same intensity to every neuron; it represents neither the number of neurons
nor the number of spikes.

Neurons remain independent instances without synaptic connections.
**Variation between neurons** uses the parameter variation and observation noise
recorded in the checkpoint. Uncheck it to view the central parameterizations.
The same seed and settings reproduce the population; **New variation** changes
the seed.

The AI receives only voltage traces, in batches of at most 64 neurons, and
classifies each independently. Expected labels are used only to compare
results. The architecture, weights, and training pipeline remain compatible
with `weights/classifier.pt`. The `.pth` files from the legacy implementation
cannot replace that checkpoint.

### Connected networks (exploratory)

In the left panel, **Synaptic connections → Connect neurons** enables the
network. **Preset: 10,000 neurons** configures 3,334 RS, 3,333 CH, and 3,333 FS
neurons with 100 outgoing connections per neuron: **1,000,000 directed synapses**.
It preserves the selected stimulus, seed, and variation setting.

- Each neuron connects to distinct random targets without self-connections.
  In small populations, outgoing connections are capped at `N − 1`.
- Each spike delivers a signed current after the configured delay. Defaults
  are excitatory strength 0.5, 20% inhibitory neurons, inhibitory gain 4,
  a 1 ms delay, and 5 ms exponential decay, in model units.
- Excitatory/inhibitory roles are sampled separately from RS/CH/FS parameters.
  They do not identify real biological cell types.
- The same seed reproduces parameters, connections, and roles. Changing only
  strength or delay preserves topology for comparisons.
- There is a **1,000,000,000-synapse** limit and at most 1,000 outgoing
  connections per neuron. Over-budget settings are rejected before network
  allocation. Both independent and connected modes allow up to 1,000,000 neurons
  total. For example: 333,333 RS, 333,333 CH, and 333,334 FS. With 100 outgoing
  connections per neuron, this population has 100 million synapses; with 1,000,
  it reaches one billion. These are configuration limits, not benchmark results
  at that scale.

The **Network** tab shows the population-wide firing rate in bins of up to
5 ms, mean synaptic current, connection counts between RS/CH/FS groups, and
the selected neuron's role, incoming and outgoing connections, and sample of
targets. All targets remain in memory. The raster displays at most 200 neurons,
while the full trace of any neuron remains accessible.

In connected mode, spikes and rates are measured before observation noise;
this noise does not generate synaptic events. Checkpoint predictions remain
available and are labeled **experimental**. Agreement with source parameters
does not validate the classifier on networks: it was trained only on independent
neurons. Retraining is unnecessary to open the update, but scientific evaluation
of the new condition requires its own data and tests, keeping original results
separate.

Equations, update order, and limitations are in [NETWORKS.md](NETWORKS.md).
This mode uses current-based synapses; it implements neither gap junctions nor
plasticity.

### Visualization and results

- **Signals:** a raster of spike times and the selected neuron's full trace.
  Raster colors represent predicted classes; gray indicates no determined
  classification. Above 200 neurons, the raster shows an evenly distributed
  visual sample, identified in its subtitle.
- **Individual classifications:** one row per neuron with expected class,
  prediction, confidence, spikes, and status. Click a row to inspect its signal.
  A hypothesis marked `(?)` remains indeterminate.
- **Statistics:** expected and predicted counts per class, a confusion matrix
  with an indeterminate column, coverage, and accuracy among classified neurons.
  Coverage is `classified / total`; accuracy is `correct / classified`.
  If no neuron is classified, accuracy is unavailable.

Every neuron contributes to inference and statistics, including those absent
from the raster preview. Individual selection provides access to any neuron's
trace. Plot downsampling preserves peaks and does not change the signal received
by the classifier.

These statistics describe the interactive simulation and do not replace
held-out test metrics. Without a checkpoint, simulation remains available,
but predictions and classification metrics are unavailable.

### Responsiveness

Simulation and inference run in a single worker separate from the interface.
Rapid control changes are coalesced; stale requests are canceled or discarded,
retaining only the latest pending configuration. While computation finishes,
charts and tables remain identified as results of the last completed simulation.
Selecting another neuron uses results in memory without simulating or
classifying the population again.

The simulator stores values from the same timestep contiguously in memory.
Noise, spike counting, and signal validation use bounded buffers; the interface
applies noise directly to population storage. Simulation checks cancellation
every 32 steps and between noise blocks, avoiding a wait for an entire obsolete
simulation.

The interface fuses BatchNorm normalization into convolutions in a dedicated
inference copy, without modifying the trainable model or checkpoint. It uses
up to eight CPU inference threads, reserving at least one logical processor
for other tasks when available. The equation, timestep, full traces, seeded
noise sequence, and checkpoint are preserved. Inference retains float32
precision; small rounding differences between parallel kernels are possible.

### Measuring capacity on your computer

In the same environment as the interface, using your trained checkpoint:

```powershell
python benchmark_population.py --neurons 3000 10000 100000 --threads 8 --output artifacts/population_benchmark.json
```

To measure the network with the local checkpoint:

```powershell
python benchmark_population.py --connected --neurons 1000 10000 --out-degree 100 --threads 8 --output artifacts/network_benchmark.json
```

The JSON records synaptic settings, actual synapses, topology storage, delivered
events, firing rates, and timings for each repetition. Use `--simulation-only`
to measure the network without classification.

The benchmark runs three measurements per size after warm-up, including
simulation, noise, inference, statistics, and raster preparation. It excludes
interface rendering and retention of the previous population. To compare CPU
usage, repeat with `--threads 1` or `--threads 4`. Without a checkpoint,
`--simulation-only` measures only simulation and result preparation; this time
does not represent the full classification cost.

After comparing results, the interface also accepts `python main_app.py --threads 4`
(or the executable with `--threads 4`) to use a suitable thread count for the
computer. Selection is automatic without this argument.

At the default 200 ms window and 0.1 ms timestep, each neuron requires 8,000 bytes
for stored voltage alone. During an update, the interface may retain both the
previous and new populations: approximately `16,000 × neurons` bytes, plus
parameters, predictions, model buffers, and interface overhead. The configured
limit does not guarantee instantaneous updates or real-time execution.

Memory formulas, assumptions about sparse and dense connections, and estimates
for a Ryzen 7 5700X3D with 32 GiB are in [CAPACITY.md](CAPACITY.md).

The interface supports dark and light themes, indicates whether a checkpoint
is loaded, and shows **correct**, **mismatch**, and **indeterminate** states.

`assets/app_icon.png` and `assets/app_icon.ico` are included in the project.
On Windows, the application also sets its own identity so it is not visually
grouped under the Python interpreter's generic icon.

## Building the Windows application

The project includes a PyInstaller configuration for a 64-bit Windows
application. Distribution uses **onedir**, which is more reliable for PyTorch,
PyQt6, and their binaries than a single executable extracted at runtime.

On Windows, activate the same virtual environment used for training and run:

```powershell
.\build_windows.ps1
```

If the PowerShell execution policy blocks the file:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1
```

The script:

1. Confirms that it is running on Windows.
2. Checks for `weights/classifier.pt`.
3. Installs application and packaging dependencies.
4. Bundles the checkpoint, icon, and executable metadata.
5. Produces `dist/IzhikevichClassifier/IzhikevichClassifier.exe`.
6. Creates `dist/IzhikevichClassifier-Windows-x64.zip` for distribution.

The `.exe` must remain alongside the `_internal` subdirectory. A computer running
this distribution does not need Python or the project's libraries installed.
Rebuild an existing Portuguese executable to obtain the English interface and
metadata; the source translation does not modify previously built packages.

Because the executable is unsigned, Windows SmartScreen or antivirus software
may request confirmation on first launch. Code configuration cannot remove this;
public distribution without that warning requires a code-signing certificate.

Stimuli without enough spikes are shown as indeterminate. This is an
observability limitation: empty signals contain insufficient information to
distinguish all parameterizations.

## Tests

```bash
python -m unittest discover -s tests -v
```

Simulation, database, population, and metrics tests use NumPy and the standard
library. Architecture and batch-inference tests require PyTorch. Interface and
worker integration tests require graphical dependencies and run with Qt's
`offscreen` platform without opening windows. They check cancellation,
discarding stale results, individual selection, and result accounting. Test
checkpoints are untrained fixtures and do not validate scientific accuracy.

## Standalone benchmark

```bash
python benchmark_models.py --samples 512 --repeats 5
```

The benchmark compares simulation cost, not biological equivalence. LIF and
Izhikevich use different equations, states, and input scales.

## Limitations to include in the final report

1. Signals are synthetic, and labels derive from known parameterizations.
2. Accuracy measures generalization within the simulated distribution, not
   real neural tissue.
3. RS, CH, and FS are simulated phenomenological patterns, not diagnoses.
4. Classification requires observable activity; neural silence may be
   intrinsically ambiguous.
5. The LIF–Izhikevich comparison does not make the two models biologically
   equivalent representations.
