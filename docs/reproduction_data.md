# Data used for paper reproduction

The release contains the 48 confirmation cases, the 240 development cases, the
39 legacy regression cases, their scoring references, and the archived first
answers. Model-visible messages, software inputs, scoring targets, answer text,
case order and training-slot order are preserved. Absolute source-file paths
in provenance metadata have portable names. The manifests record both the
public projection hashes and the corresponding original hashes.

## Evaluation and training materials

| Material | Released location | Role |
| --- | --- | --- |
| Confirmation 48 | `data/confirmation/` | Separate frozen follow-up evaluation, including three sets of first answers |
| Bound single-request example | `data/workflow_example/` | Original public case, R6 first answer, completed model binding and protocol for the supplementary candidate table |
| Development 240 and legacy 39 | `data/evaluation/` | Public inputs and scoring references; historical base/R1/R6 answers in `historical/` |
| Continuation A/B first answers | `data/retention/arm_A/`, `arm_B/` | Original dev and legacy generations, plus case scores |
| Continuation domain rows | `data/training/domain_rows.jsonl.gz` | Fixed domain examples for A/B |
| Continuation replay 819 | `data/training/replay_rows.jsonl.gz` | Researcher-authored extraction tasks using limited source facts |
| Historical R6 training 8,560 | `data/training/r6_train.jsonl.gz` | 8,212 synthetic contract cases and 348 existing computational-result cases |
| Training slots | `data/training/slots.csv` | Exact 4,096 visits for each continuation arm |

`scripts/prepare_training_data.py` extracts the three training files and slots
into a new directory and verifies their hashes. `training/protocol.json`
pins the public files and retains the original domain/replay file digests.
The public metadata projection does not change tokenized training messages.

The released checkpoints allow all reported model decision scores to be
regenerated without reconstructing upstream R1/R5 training. Both continuation
arms can also be trained from the released R6 parent. The original complete
four-database collection and upstream adapter training runs are not redistributed
or reconstructed here.

## Source attribution and terms

The project-specific tasks and annotations are distinct from the source facts.
Database-derived records retain source names and identifiers; the 819-row replay
source list is in `data/training/replay_sources.json`. These small projections
contain factual sequence, identifier, numeric and category fields needed for the
published research tasks. They do not constitute the full source databases.
MAVEN-Pep's research-use license does not relicense third-party material.

* [DBAASP](https://dbaasp.org/) supplies peptide identities and assay fields.
  Its [terms](https://dbaasp.org/terms-and-conditions) require attribution when
  material is copied, adapted or redistributed. Database credit:
  [Pirtskhalava and colleagues, DBAASP v3 (2021)](https://doi.org/10.1093/nar/gkaa991).
* [DRAMP](https://dramp.cpu-bioinfor.org/) supplies peptide identities and
  annotated properties under its published CC BY 4.0 notice. Cite DRAMP and
  retain applicable source terms, including its original-article and
  patent-related permissions notice. Database credit:
  [Ma and colleagues, DRAMP 4.0 (2025)](https://pubmed.ncbi.nlm.nih.gov/39526377/).
* [Propedia v2.3](https://github.com/LBS-UFMG/propedia) supplies peptide–protein
  complex identifiers and structural facts. Cite Martins and colleagues'
  [2021 description](https://doi.org/10.1186/s12859-020-03881-z) and
  [v2.3 update](https://doi.org/10.3389/fbinf.2023.1103103).
  The historical v2.3 source is identified explicitly; a later version's license
  is not represented as a license for the historical complete collection.
* [SATPdb](https://webs.iiitd.edu.in/raghava/satpdb/) supplies peptide identifiers,
  sequences and category facts. Cite Singh and colleagues'
  [database description](https://doi.org/10.1093/nar/gkv1114).

Use the original sources and their terms when obtaining larger database
collections. The released task fixtures do not supply new experimental peptide
activity measurements.

## Generation and scoring

Generation and scoring run as separate commands. Generation consumes public
messages and verifies the model, tokenizer and input identities before producing
new first answers. It does not open the scoring references. The scorer requires
one answer per case, verifies model identities, and uses the original executable
contracts and legacy field scorer.

Historical dev/legacy generations used two index-modulo ranks. Run rank 0 and
rank 1 sequentially on one GPU, or concurrently with two GPUs, to preserve the
original four-case batches. Continuation A/B generations use one sequential
rank each. These partition choices matter even with greedy decoding because
floating-point kernels can depend on batch shapes.

The confirmation and continuation runs used the released CUDA 12.6 environment.
The older dev/legacy three-model records used PyTorch 2.5.1+cu124 and Python
3.12.3; their version metadata is retained in `data/evaluation/manifest.json`.
Use archived first answers for an exact CPU re-score. A new inference run records
its own software versions and must be compared with the published scores;
different numerical runtimes are not assumed to produce identical text.
