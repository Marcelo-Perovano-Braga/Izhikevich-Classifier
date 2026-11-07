# Protocol 2: varied envelopes and group-based selection

The first expanded candidate failed on the triangular stimulus. That result
became development information. Triangles and double pulses are no longer
unseen families in this protocol.

```bash
python train_robust.py --protocol 2 --seed 27149 --epochs 50 --patience 12 --train-per-class 200 --validation-per-class 50 --test-per-class 100 --threads 4 --adopt-if-superior
```

`--protocol 1` remains available to reproduce the previous study. When comparing
with `--candidate`, the version is read from the checkpoint itself.

## Candidate fitting

The `NeuronClassifier1D` architecture and 0.7 confidence threshold are retained.
The model is initialized from scratch. The previous 25 controls are preserved,
with 31 additional conditions: descending ramps, triangles, double pulses,
random pulses, random steps, interpolated envelopes, random ramps, random
asymmetric triangles, and random sinusoids. Four of these families are also
simulated in connected networks. Random envelopes vary per neuron, using an
RNG stream separate from labels, parameters, amplitudes, and observation noise.

With the settings above: 33,600 training traces, 8,400 selection traces, and
8,400 calibration traces. Classes are balanced within each condition. All
neurons are retained, including silent ones. Fitting namespaces: 30, 31, and 32.

An epoch with reference accuracy of at least 99% on the selection split takes
priority over epochs below this floor. Among eligible epochs, minimize:

`mean NLL + 0.5 × worst per-condition NLL + 0.25 × worst per-condition/class NLL`.

When no epoch is eligible, add `10 × reference accuracy shortfall` to the score
and record the best provisional result. Final comparison still requires the
original 99% floor. The calibration set is used only to fit a scalar temperature
after selecting the weights.

## Confirmation reserved before training

`training/reserved_comparison.json` records conditions, policy, namespace, and
seeds before generating fitting data. Its SHA-256 is included in the checkpoint.
The comparison rejects changes to this reservation and traces that exactly
match any split used for fitting or calibration.

There are 30 conditions, five new seeds (`7109, 7211, 7307, 7411, 7523`), namespace
33, and 100 neurons per class/condition/seed: 45,000 signals per model. Both
models receive the same signals. Three families are excluded from every fitting
split: a 5–65 Hz chirp, a descending exponential envelope, and a raised-cosine
burst. Each is tested with 0.2 and 5 mV noise. These are held-out generator
families; there is no claim of mathematical independence from every combination
of the other waveforms.

The ten adoption criteria referenced in `ROBUST_TRAINING.md` remain unchanged,
including the three-percentage-point regression limit for any condition.
There is no threshold tuning or weight reselection after observing confirmation.
If any criterion fails, the active checkpoint remains intact. After inspection,
this confirmation becomes development information for future iterations;
a new confirmation must be reserved.

The five seeds are the simulation replicates. Connected neurons are not
statistically independent replicates. Labels identify RS/CH/FS parameter
families, not biological diagnoses. Engineering criteria in synthetic simulation
do not demonstrate operation on real neural signals.
