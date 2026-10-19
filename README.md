# MAVEN-Pep

Code, frozen checkpoints and data for the MAVEN-Pep manuscript on locally
deployable modular C5a peptide prioritization. The mechanism produces a
researcher-defined priority score, not an activity probability. Experimental
feedback learning remains future work.

Author-created code and LoRA weights are licensed **only for non-commercial
academic research and paper reproduction**; see [LICENSE](LICENSE). Third-party
licenses and database attribution remain applicable.

## Reproduce the numerical results without a GPU

Use Python 3.11 or newer from the repository root. These commands use the Python
standard library and the released first answers, without a model or API key.

```bash
python -B scripts/offline_selection.py --validate
python -B scripts/compare_panels.py --output-dir reproduced/mechanism_panel
python -B scripts/reproduce_model_results.py --output-dir reproduced/recorded
python -B -m unittest discover -s tests -v
```

| Result | Recorded value |
| --- | --- |
| Construct top ten | 4 distinct sequences; mean priority 0.9315012386 |
| Sequence top ten | 10 distinct sequences; mean priority 0.9226762287 |
| Confirmation semantic success | Qwen3-8B 22/48; R1 0/48; R6 33/48; public rules 40/48 |
| Development strict success | Qwen3-8B 112/240; R1 0/240; R6 225/240 |
| Legacy strict success | Qwen3-8B 32/39; R1 35/39; R6 30/39 |
| Continuation A, domain only | 225/240 development; 31/39 legacy |
| Continuation B, 20% replay | 226/240 development; 30/39 legacy |

The 48-case confirmation is separate from the development cases. Public rules
use public input only. The privileged structured scorer's 48/48 is an upper-bound
check using normalized scoring facts, not an independent language-model result.
The three-model supplement is not the originally proposed eleven-model panel.

## Download the exact models

[Model release](https://github.com/weizesunmd-byte/MAVEN-Pep/releases/tag/models-v1)
contains R1, R6, continuation A, continuation B and the historical tokenizer.
The four adapters contain 43,646,976 trainable LoRA parameters each. Their weight
bytes match the checkpoint identities recorded with the experiments.

```bash
python -B scripts/download_models.py --include-base
python -B scripts/verify_models.py
```

The downloader obtains adapters and tokenizer from this project's GitHub Release,
and the 16.4 GB Qwen3-8B foundation model from its official publisher at revision
`b968826d9c46dd6066d109eabc6255188de91218`. It verifies file sizes and SHA-256,
retains partial downloads for resumption, and preserves existing verified files.
The checksums and fixed URLs are in `assets/model_release.json`.

For only the core R6 comparison, use
`--components r1 r6 --include-base`. The tokenizer is always included. Files go
into ignored `models/`; `--output-dir` selects another location.

## Regenerate model answers on a GPU

Use a CUDA-compatible PyTorch build for your hardware, then install
`requirements-gpu.txt`. The confirmation and continuation records used PyTorch
2.14.0+cu126, CUDA 12.6, Transformers 4.57.6 and PEFT 0.18.1 on a 32 GB Tesla
PG500-216. Use FP16 and SDPA; verify support for your GPU architecture.

```bash
python -m pip install -r requirements-gpu.txt
python -B scripts/infer_confirmation.py --model base --output-dir reproduced/new_confirmation
python -B scripts/infer_confirmation.py --model old --output-dir reproduced/new_confirmation
python -B scripts/infer_confirmation.py --model candidate --output-dir reproduced/new_confirmation
python -B scripts/reproduce_model_results.py --predictions-dir reproduced/new_confirmation --output-dir reproduced/new_confirmation_scores
```

Generation uses public cases only, verifies weights before model allocation, and
refuses to overwrite first answers. It retains greedy decoding, disabled
thinking, four-case batches and the 480-token output limit. Scoring is a separate
CPU command and checks the reported success counts and paired bootstrap.

For continuation A/B, generate and score each arm separately:

```bash
python -B scripts/evaluate_retention.py --mode generate --model A --output-dir reproduced/new_A
python -B scripts/evaluate_retention.py --mode score --model A --prediction-dir reproduced/new_A --output-dir reproduced/new_A_scores --check-recorded
python -B scripts/evaluate_retention.py --mode generate --model B --output-dir reproduced/new_B
python -B scripts/evaluate_retention.py --mode score --model B --prediction-dir reproduced/new_B --output-dir reproduced/new_B_scores --check-recorded
```

For the historical dev/legacy comparisons, preserve the original two-rank batch
partition. These two commands may run sequentially on one GPU. Repeat with
`--model base` and `--model old` for the other comparisons.

```bash
python -B scripts/evaluate_retention.py --mode generate --model candidate --rank 0 --world-size 2 --output-dir reproduced/new_R6_development
python -B scripts/evaluate_retention.py --mode generate --model candidate --rank 1 --world-size 2 --output-dir reproduced/new_R6_development
python -B scripts/evaluate_retention.py --mode score --model candidate --prediction-dir reproduced/new_R6_development --output-dir reproduced/new_R6_development_scores --check-recorded
```

The older three-model dev/legacy records used PyTorch 2.5.1+cu124. Different
GPU/software kernels can change generated text; new runs record their environment
and are explicitly compared with the historical results. The included first
answers support exact CPU re-scoring in any case.

## Reproduce continuation training

Both fixed-budget arms start from the released R6 checkpoint. All domain,
replay and slot records needed by the two arms are included.

```bash
python -B scripts/prepare_training_data.py --output-dir reproduced/training_data
python -B training/train_arm.py --mode validate --arm A --data-dir reproduced/training_data --tokenizer models/tokenizer --output-dir reproduced/validate_A
python -B training/train_arm.py --mode validate --arm B --data-dir reproduced/training_data --tokenizer models/tokenizer --output-dir reproduced/validate_B
python -B training/train_arm.py --mode train --arm A --data-dir reproduced/training_data --tokenizer models/tokenizer --base-model models/Qwen3-8B --parent-adapter models/r6 --output-dir reproduced/training_A
```

Use `--arm B` and a new output directory for the replay condition. Each run uses
128 optimizer updates, global batch 32, the fixed learning-rate schedule and
assistant-only semantic supervision. Validation tokenizes every selected visit
and checks loss masks without allocating the model. The public port preserves
the calculation; it is not the historical worker byte sequence.

The 8,560-row R6 training corpus is included for inspection. Reconstructing the
earlier R1/R5 training lineage and the complete original four-database snapshot
is not required for result reproduction from the released checkpoints and is
not reconstructed by this package.

## Figures and data

```bash
python -m pip install -r requirements.txt
python -B scripts/make_figures.py --output-dir reproduced/figures
python -B scripts/check_ties.py --output reproduced/tie_sensitivity.csv
python -B scripts/check_monte_carlo.py --output-dir reproduced/monte_carlo
```

The three quantitative figures are plotted from released tables. The two
conceptual workflow illustrations are provided in the manuscript and supplement.
See [data and source attribution](docs/reproduction_data.md) for the evaluation
inputs, scoring references, database-derived fact projections and upstream terms.
No credentials or full third-party database dumps are included.
