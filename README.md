# DDKG — Dynamic Dual Knowledge Graph captioning (experiment package)

Single-GPU (RTX 4070, 12 GB) pipeline for the DDKG paper: cached Faster R-CNN region features → scene graph +
ConceptNet concept graph → Dual Graph Encoder → caption-gated Transformer decoder, with ablations, multi-seed runs,
official coco-caption metrics and paired significance tests.

> **Status: written without being able to execute PyTorch** (the authoring sandbox had no GPU, no torch, no network).
> Only the pure-Python parts were run (config, vocab, Karpathy loader, feature store, experiment runner, aggregator,
> static import/config-key checks). **The model code has never been executed.** Step 2 below (`smoke_test.py`) is
> the first real test; expect to fix a few errors. No result in this package or in the paper comes from a real run yet.

## What changed vs. the first draft (bugs fixed by design)
| First-draft problem | Fix here |
|---|---|
| Edge scorer and retrieval projection received no gradient; sparsity loss used detached floats | Edges are **soft weights** (`sigmoid`), added as log-bias to GAT logits; pruner score multiplies the weight; sparsity loss = mean soft weight (differentiable). Retrieval similarity of the *selected* concepts is recomputed with gradients and fed to the gate MLP. |
| Edge set of image 0 shared by the whole batch | Dense **per-image** adjacency `[B,N,N]` |
| Padded nodes attended to | One additive float mask everywhere (own `MHA` on `scaled_dot_product_attention`); self-test proves padding invariance |
| Gate multiplied concept vectors then LayerNorm erased it | Gate enters as **attention log-bias** + a learned gate embedding |
| "Dynamic" gate used pooled image only | **Caption-conditioned gate** per decoder layer and token (`model.caption_gating`) |
| Edge threshold 0.3 on a 100-way softmax → empty graphs at init | Separate edge logit, init ≈ dense graph, pruned progressively |
| Detector in the training loop, RoI feats zero-padded | Features **extracted once** (`extract_features.py`), true 1024-d box-head features |
| Random fallback KG silently used | `knowledge_graph.require_real: true` refuses to run without ConceptNet |
| Own metric re-implementations | **Official `pycocoevalcap`** for final numbers; result files record whether the run was fully official |

## 0. Install
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
java -version                                              # Java 8+ needed for PTBTokenizer/METEOR/SPICE
```
`pycocoevalcap` downloads Stanford CoreNLP jars for SPICE on first use (needs internet once).

## 1. Data
```text
/data/karpathy/dataset_coco.json          # caption_datasets.zip from cs.stanford.edu/people/karpathy/deepimagesent
/data/coco/{train2014,val2014}/...        # COCO 2014 images
/data/kg/                                  # created in step 1c
```
Edit paths in `configs/base.yaml` (or pass `--set data.image_root=...`).

**1a. Vocabulary:** `python scripts/prepare_data.py`
**1b. Region features (once, resumable):** `python scripts/extract_features.py`  (COCO ≈ 9 GB fp16; time not measured — run with `--splits val` first to gauge it)
**1c. ConceptNet:** download `numberbatch-en-19.08.txt.gz`, then
`python scripts/prepare_conceptnet.py --input numberbatch-en-19.08.txt.gz --output-dir /data/kg --vocab-json /data/features/coco/vocab.json`

## 2. Verify the code on your machine (no data needed)
```bash
python scripts/smoke_test.py            # all 8 variants: loss, backward, gradient-reaches-module checks, beam search,
                                        # padding invariance, decoder causality; + bf16 autocast on GPU
pytest tests -q
```
If something fails, send the full traceback; do not start long runs until this passes.

## 3. Train one run
```bash
python scripts/train.py --config configs/base.yaml --set training.epochs=2 --set run.out_dir=results/dbg   # quick check
python scripts/train.py --config configs/base.yaml
```
Outputs in `run.out_dir`: `config.json`, `train.log`, `history.json`, `best.pt`, `last.pt` (auto-resume),
`test_metrics.json`, `test_predictions.json`, `test_per_image_cider.json`, `run_summary.json` (parameters, peak GPU memory, time).
Model selection: best validation CIDEr (first `eval.val_images` val images, greedy). Final: beam 3 on the full Karpathy test split.

## 4. Experiments → paper tables
```bash
python scripts/run_experiments.py --suite main     --seeds 0 1 2 --dataset coco        # Table 1 (baseline vs DDKG)
python scripts/run_experiments.py --suite ablation --seeds 0 1 2 --dataset coco        # Section 5 table
python scripts/run_experiments.py --suite main --seeds 0 1 2 --dataset flickr30k \
   --set data.karpathy_json=/data/karpathy/dataset_flickr30k.json --set data.image_root=/data/flickr30k/images \
   --set data.feature_root=/data/features/flickr30k --set data.vocab_path=/data/features/flickr30k/vocab.json
python scripts/aggregate_results.py --dataset coco       # -> results/coco_tables.md, _summary.json, _table.tex
```
`--dry-run` prints the commands; finished runs are skipped, so the command can be re-issued after an interruption.
Run time is unmeasured: take it from `run_summary.json` after the first run, then decide how many seeds/epochs you can afford
(`--set training.epochs=...`). The ablation suite is 10 experiments × seeds.

| Paper section | Experiment names (`configs/experiments.yaml`) |
|---|---|
| Table 1/2 baseline | `baseline_transformer` (region Transformer encoder, same features/decoder, no graphs/KG) |
| Ablation: knowledge | `no_kg`, `static_gate` |
| Ablation: caption-conditioned gate | `image_only_gate` vs `full` |
| Ablation: encoder parts | `no_cross_attn`, `no_edge_pruning`, `dense_kg_attention` (also gives the measured memory vs grouped) |
| Sensitivity to K | `k5`, `k10`, `k40` |

## Known limitations (state them in the paper)
- **Cross-entropy training only** (no SCST/CIDEr optimisation), so absolute scores will be below SCST-trained SOTA; compare against baselines trained the same way.
- `baseline_transformer` is *our* same-feature baseline, not the published BUTD/AoANet/Oscar numbers; quote those from their papers with citation, flagged as different training regimes.
- Relation types are latent (no Visual Genome relation supervision); edge existence is learned only from the caption loss.
- Concept retrieval is weakly supervised (through the gate); no grounding loss to detector classes.
- Numberbatch file is not frequency-sorted; concept vocabulary = caption-vocab words first, then file order (see `prepare_conceptnet.py`).
- Features use min image side 600 (speed); results are not comparable with 800px extractions.
- Metrics are official only if `official: true` appears in `test_metrics.json` (Java available).
