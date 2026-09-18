# Urdu long-line UHWR training

Copy these four new files into your training machine's `uhwr-icdar-main`:

- `long_line_model.py`
- `train_long_lines.py`
- `check_long_lines.py`
- `LONG_LINE_TRAINING.md`

Optional local subset-selection tests: `test_periodic_subset.py`.

Existing UHWR model/tokenizer files are reused and not edited. The temporary EMURU implementation was removed and its model changes reverted. Do not copy the abandoned EMURU trainer/config.

## Architecture and initialization

The model retains the mixed model's CNN, 256-dimensional three-layer/eight-head RoBERTa encoder, three-layer/eight-head GPT-2 decoder, 512 learned decoder positions, and original byte tokenizer. It uses the learned decoder position convention in `train_mixed.py`. No external decoder-pretraining directory is needed when initializing from the complete mixed checkpoint.

The tokenizer is constructed directly from the existing vocabulary/merges with UHWR special tokens supplied immediately. This avoids adding GPT-2's default `<|endoftext|>` token: the supplied checkpoint has 261 decoder rows and 262 CTC rows (including blank).

The custom visual position table becomes 2048 rows. RoBERTa's position table includes its offset (2050 rows with the current padding index). Existing visual position rows are copied; additional rows are initialized. Every other parameter is loaded strictly, so an incompatible checkpoint raises an error. The CTC head is loaded and frozen but does not contribute a loss. The whole CNN/encoder/decoder is fine-tuned with cross-entropy and 0.1 label smoothing.

Images remain 64 pixels tall and retain their width, up to 8192. Transposition, pixel inversion, and the original vertical flip are retained for checkpoint compatibility; this flip is not described as RTL reversal. There is no crop or width compression. Images are dynamically padded with white background (zero after inversion), and valid CNN lengths mask encoder and decoder cross-attention. Greedy decoding uses a KV cache, stops at EOS, and permits up to 511 new tokens after BOS.

## Setup

Run in your existing activated training environment:

```bash
export UHWR_REPO="/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/uhwr-icdar-main"
export URDU_SHARDS="/home/ayeshaamjad/Desktop/PtHW_gen/Emuru-autoregressive-text-img/DATA_GEN/Urdu_Corpus/dataset/webdataset_shards_shuffled_final"
# This path is also the default in both Python scripts:
export INIT_WEIGHTS="/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/best_model_mixed.pt"
cd "$UHWR_REPO"
python3 -c 'import torch, transformers, webdataset, wandb, jiwer, torchvision; print(torch.__version__, transformers.__version__); print(torch.cuda.get_device_name(0))'
wandb login
```

The implementation targets the repository's torch 2.7.1, transformers 4.54.0, webdataset 1.0.2, wandb 0.21.0, and jiwer 4.0.0. First use your existing environment. If missing, install only missing dependencies; avoid replacing your CUDA-compatible PyTorch unnecessarily.

## Check checkpoint and full-width forward passes

```bash
python3 check_long_lines.py --init_checkpoint "$INIT_WEIGHTS"
```

Checks tokenizer round-trip, preservation of checkpoint position weights, 256/7188/8192-pixel forward passes, cached generation, unequal-width masks, and finite CNN gradients. It uses synthetic images and does not train on your dataset.

## Smoke run

```bash
python3 train_long_lines.py \
  --data_dir "$URDU_SHARDS" \
  --init_checkpoint "$INIT_WEIGHTS" \
  --output_dir results_uhwr_smoke \
  --epochs 1 --max_batches 64 --eval_samples 32 \
  --eval_every_steps 4 --save_every_steps 4 --periodic_eval_samples 16 \
  --batch_size 8 --pixel_budget 8192 --accumulation 8 \
  --workers 2 --precision fp16 \
  --wandb_name urdu-long-lines-smoke
```

Inspect finite loss and W&B images/references/predictions. A smoke run establishes operational compatibility, not accuracy. Use a separate output directory for actual training.

## Fine-tune

```bash
python3 train_long_lines.py \
  --data_dir "$URDU_SHARDS" \
  --init_checkpoint "$INIT_WEIGHTS" \
  --output_dir results_uhwr_long \
  --epochs 20 --eval_samples 2000 \
  --batch_size 8 --pixel_budget 8192 --accumulation 8 \
  --workers 4 --precision fp16 --lr 0.0001 \
  --wandb_name urdu-long-lines-finetune
```

The fixed seeded shard split has 220 train, 12 validation, and 13 test shards. `splits.json` records the exact filenames and is reused on resume/test. Shards are shuffled before assigning splits. This is a sample-level synthetic split: duplicate transcriptions and fonts can occur across splits; it is not an unseen-text or unseen-font benchmark.

The bounded streaming queues group similar widths. The width buckets are 256/512/1024/1600/2048/3072/4096/6144/8192. Batch size is bounded by `min(batch_size, pixel_budget // bucket_width)` with a minimum of one, so widths over 4096 use batch size one with the default budget. Padding is to the actual maximum batch width, rounded to eight pixels. No TAR extraction is needed.

An inference baseline runs before training. By default, every 500 optimizer steps the script independently decodes 128 fixed validation images: 32 each with widths <=1024, 1025–2048, 2049–4096, and >4096. The subset is cached in `periodic_eval/manifest.json` and PNG files. Selection scans validation shards until quotas are met; if a group has insufficient samples, the actual smaller count is reported. This width-balanced subset intentionally overrepresents long lines and is a diagnostic rather than a corpus-wide CER estimate.

`periodic_val/cer`, `periodic_val/wer`, `periodic_val/exact_match`, width-specific CER, missing-EOS rate, substitution/deletion/insertion rates, and longest-image prediction tables appear in W&B. Predictions are also written to `periodic_predictions.jsonl`, replaced each evaluation. `best_periodic` is selected on this fixed subset; `best` remains selected on epoch-end validation. The LR scheduler continues to use epoch-end validation only.

Use `--eval_every_steps 500 --periodic_eval_samples 128 --save_every_steps 500` to make the defaults explicit; `--eval_every_steps 0` disables periodic validation. Keep the subset size unchanged for an existing output directory. Every 500 steps, an atomic `last.pt` save records optimizer state and completed microbatch count at an accumulation boundary.

Epoch-end validation independently decodes the first 2000 samples of the deterministic validation stream each epoch. Use `--eval_samples 0` for all validation samples, at higher runtime. Best checkpoint selection and the plateau LR scheduler minimize this inference CER. Validation loss is separately teacher-forced. Test always evaluates all held-out test shards. These synthetic tests do not establish real-handwriting performance.

W&B logs training loss, gradient norm, learning rate, padded width, batch size, peak allocated VRAM, validation CER with spaces, WER, exact match, missing-EOS rate, CER by width, and examples of the longest validation images with references/predictions. Microbatch losses are averaged over the actual accumulation count; effective sample count varies with width.

If CUDA runs out of memory, first use `--batch_size 4 --pixel_budget 4096`. If a single long line still fails, further architectural/memory changes are needed; do not silently resize or crop it. If the GPU supports BF16, `--precision bf16` is available. Only a GPU run can establish the actual memory requirement.

## Resume

```bash
python3 train_long_lines.py \
  --data_dir "$URDU_SHARDS" \
  --output_dir results_uhwr_long --resume \
  --epochs 20 --eval_samples 2000 \
  --batch_size 8 --pixel_budget 8192 --accumulation 8 \
  --workers 4 --precision fp16 --lr 0.0001
```

`last.pt` contains model/optimizer/scheduler/scaler/RNG states and the W&B run ID. An epoch-end checkpoint resumes at the next epoch. A periodic checkpoint resumes inside the epoch by deterministically replaying the data stream and skipping consumed batches; this replay can take time. Worker count, batching, seed, precision, accumulation, label smoothing, and smoke limits must match the saved run for mid-epoch resume. Updates since the last checkpoint are not saved. Earlier epoch-end checkpoints remain compatible; use the same training settings when resuming. A new run cannot overwrite an output directory containing `last.pt`.

## Held-out test

```bash
python3 train_long_lines.py \
  --mode test --data_dir "$URDU_SHARDS" \
  --output_dir results_uhwr_long \
  --checkpoint results_uhwr_long/best \
  --batch_size 8 --pixel_budget 8192 --precision fp16 \
  --wandb_name urdu-long-lines-heldout-test
```

Outputs `test_metrics.json` and `test_predictions.jsonl` (key, width, reference, prediction). Test uses the saved split; do not change it after assessing results.

Saved model folders use this implementation's `model.pt` and `config.json`, rather than Hugging Face's generic `from_pretrained` format. Load with `LongLineModel.from_pretrained(path)`.

## Local verification limits

Source syntax was checked in the development workspace. PyTorch, Transformers, and WebDataset are not installed there, and the dataset/GPU are on your training machine; runtime model and data checks must be run there with the commands above. No dataset training was launched in the development workspace.

## Switching your currently running older process

Copy the updated `train_long_lines.py` only after the old process completes its current epoch and writes `results_uhwr_long/last.pt`. The running process cannot adopt the new code. Stop it after that checkpoint is confirmed, then copy the update and run the resume command above. Any updates after that saved checkpoint will be replayed. Do not interrupt an older process before its first checkpoint unless you accept losing unsaved progress. No architecture or tokenizer changes are required for this update.

For a fresh run, periodic validation/checkpointing are enabled automatically. For an existing run, `--resume` loads the saved state and immediately logs a fixed-subset baseline before continuing. Model mode returns to training after each periodic evaluation, and evaluation runs only at optimizer boundaries so pending gradients are not discarded.

Development checks for this update: Python syntax and five dependency-light tests covering balanced selection, width boundaries, cache reuse, changed configuration, missing cached images, and insufficient samples. Full GPU periodic evaluation/resume must be verified on the training machine.
