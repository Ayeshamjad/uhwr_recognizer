# Compare the original and fine-tuned UHWR on short inputs

Copy `compare_short_htr.py` into your training machine's `uhwr-icdar-main`. Existing training/model files are not edited. Optional dependency-light checks are in `test_short_comparison.py`.

This script loads checkpoints for inference only. It never trains, updates checkpoints, changes splits, or sends W&B messages. It saves local predictions, images, and metrics. It uses GPU resources, so running alongside training can reduce training throughput. `--device cpu` is available but slower.

## Six controlled comparisons

| Variant | Weights | Image handling | Decoder |
| --- | --- | --- | --- |
| `original_beam4` | Original mixed checkpoint | Original 256/512/1024/1600 canvas and no source mask | Four-beam, matching mixed evaluation settings |
| `original_greedy_control` | Original mixed checkpoint | Same original handling | Greedy, PAD/BOS suppressed as in the new pipeline |
| `modified_original_greedy` | Original checkpoint with extended position tables | Full width, source masks | Current greedy method |
| `modified_original_beam4` | Same original checkpoint | Full width, source masks | Four-beam |
| `finetuned_greedy` | Current fine-tuned checkpoint | Full width, source masks | Current greedy method |
| `finetuned_beam4` | Current fine-tuned checkpoint | Full width, source masks | Four-beam |

Original preprocessing reproduces `train_mixed.py::MixedHWRDataset._preprocess` with augmentation disabled: grayscale, aspect-ratio fit, inversion, transposition, and vertical flip. The original model uses no source padding mask. Learned decoder positions are loaded from the checkpoint, overriding the sinusoidal builder's initial values.

All metrics preserve spaces and use independent decoding. Original beam evaluation did not suppress special tokens; new decoders do. The original greedy control uses the same suppression as the modified pipeline so their comparison focuses on image handling/masks/position capacity. Old vocabulary IDs are preserved by the corrected `Tokens` helper. Do not import/run `train_mixed.py` just to build the model: its startup patch is unnecessary when loading all checkpoint parameters for inference.

## Option A: rendered validation examples, executable now

If original evaluation images are unavailable, start here. The script tries to select 10 words and 10 short lines from validation ONLY. It excludes images wider than 1024 after height normalization and texts longer than 100 characters. If the validation split has fewer qualifying examples, the actual counts are printed. This is not a reconstruction of the original evaluation dataset.

```bash
export UHWR_REPO="/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/uhwr-icdar-main"
export URDU_SHARDS="/home/ayeshaamjad/Desktop/PtHW_gen/Emuru-autoregressive-text-img/DATA_GEN/Urdu_Corpus/dataset/webdataset_shards_shuffled_final"
cd "$UHWR_REPO"
python3 compare_short_htr.py \
  --data_dir "$URDU_SHARDS" \
  --split_manifest results_uhwr_long/splits.json \
  --finetuned_checkpoint results_uhwr_long/last.pt \
  --count 20 --output_dir short_htr_comparison
```

The original checkpoint path defaults to `/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/best_model_mixed.pt`. `last.pt` is atomically replaced by our training script; the comparison loads one fixed snapshot and records its optimizer step in `metadata.json`. Only supply trusted `last.pt` files produced by your trainer, because loading resumable RNG state requires `weights_only=False`. The original checkpoint is loaded using `weights_only=True`.

## Option B: original words/short sentences with exact labels

Create a UTF-8 CSV with columns `file_name,text`, quoting text containing commas. Example structure (replace these filenames/labels with actual examples):

```csv
file_name,text
word_01.png,اردو
line_01.png,یہ ایک مختصر جملہ ہے
```

Relative image paths resolve under `--image_root`, or under the CSV's directory if that flag is omitted. Choose examples independently of the predictions; ideally use original held-out samples. Dataset provenance is your responsibility: we do not know whether arbitrary external images were in the original checkpoint's training set.

```bash
python3 compare_short_htr.py \
  --samples_csv /actual/path/to/short_labels.csv \
  --image_root /actual/path/to/images \
  --finetuned_checkpoint results_uhwr_long/last.pt \
  --count 20 --output_dir original_short_comparison
```

External images of other heights are fit to height 64 with aspect ratio preserved in the modified pipeline. Original handling fits its bucket canvas using the exact original OpenCV resize. Therefore the external-image comparison also includes any interpolation/height-normalization difference. Shard images are already height 64, so they receive no height resize in the modified pipeline.

## Outputs and interpretation

- `metrics.json`: corpus CER/WER, exact-line accuracy, EOS failures, error counts, and separate word/short-line results for every variant.
- `predictions.jsonl`: reference and each variant's output, per-example CER, EOS flag.
- `samples.csv`, `samples.json`, and PNGs: exact selected labeled inputs for repeatable comparisons.
- `metadata.json`: invocation, sample source, and fine-tuned checkpoint snapshot step when loading `last.pt`.

To repeat the SAME images after more training:

```bash
python3 compare_short_htr.py \
  --samples_csv short_htr_comparison/samples.csv \
  --finetuned_checkpoint results_uhwr_long/last.pt \
  --count 20 --output_dir short_htr_comparison_later
```

Useful findings:

- Original greedy good, modified original greedy poor: investigate padding/source masks and input handling before attributing everything to domain mismatch.
- Original beam better than original greedy: decoding contributes to the discrepancy. Confirm with the modified beam/greedy comparison too.
- Both original pipelines poor on rendered validation: shows poor transfer on those examples, not that the remembered original-dataset score was wrong.
- Original pipeline good on genuine original evaluation samples: helps reproduce original behavior; compare the modified pipeline on those same samples next.
- Fine-tuned model improves rendered inputs but worsens original inputs: possible domain specialization; determine whether preserving the original domain is a project requirement.

Do not infer a reliable 90% benchmark from only 20 selected examples, and do not equate exact-line accuracy with character accuracy. This diagnostic identifies differences worth testing on a larger held-out set.

## Verification

Development syntax and five preprocessing/CSV-selection tests passed. GPU model execution and actual checkpoint predictions cannot be run in the development workspace; run the comparison on the training machine and share `metrics.json` plus a few prediction rows.
