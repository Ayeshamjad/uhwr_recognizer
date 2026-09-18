"""Single-GPU, full-line Urdu UHWR training and held-out evaluation."""
import argparse
import contextlib
import json
import math
import random
import time
import warnings
from pathlib import Path

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision.transforms.functional import pil_to_tensor

from transformers import GPT2Tokenizer
from long_line_model import LongLineModel


class Tokens:
    def __init__(self, path):
        path = Path(path)
        # Supply UHWR specials at construction time. Loading GPT-2 defaults
        # first can add <|endoftext|>, although mixed training used vocab_size
        # (261) rather than len(tokenizer) (262) for its model heads.
        vocabulary = json.loads((path/'vocab.json').read_text())
        self.tokenizer = GPT2Tokenizer(
            vocab_file=str(path/'vocab.json'), merges_file=str(path/'merges.txt'),
            bos_token='<s>', eos_token='</s>', pad_token='<pad>',
            unk_token='<unk>', mask_token='<mask>',
        )
        if self.tokenizer.get_vocab() != vocabulary:
            raise ValueError('Tokenizer added entries outside the checkpoint vocabulary')
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError('Vocabulary IDs must be contiguous from zero')
        self.sos, self.eos, self.pad = (self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.tokenizer)

    def encode(self, text):
        return torch.tensor(self.tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)

    def decode(self, tokens, stops=None):
        result = []
        for row in tokens.cpu().tolist():
            if self.eos in row:
                row = row[:row.index(self.eos)]
            result.append(self.tokenizer.decode(row, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        return result


def collate(samples, alphabet):
    widths = torch.tensor([s[0].size(-1) for s in samples])
    width = math.ceil(int(widths.max()) / 8) * 8
    images = torch.stack([F.pad(s[0], (0, width - s[0].size(-1)), value=0) for s in samples])
    labels = [torch.cat((torch.tensor([alphabet.sos]), alphabet.encode(s[1]), torch.tensor([alphabet.eos]))) for s in samples]
    return {'images': images, 'widths': widths,
            'labels': torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=alphabet.pad),
            'texts': [s[1] for s in samples], 'keys': [s[2] for s in samples]}


class Lines(IterableDataset):
    def __init__(self, shards, alphabet, train, seed, batch_size, pixel_budget, max_width):
        self.shards, self.alphabet, self.train = shards, alphabet, train
        self.seed, self.batch_size = seed, batch_size
        self.pixel_budget, self.max_width = pixel_budget, max_width

    def __iter__(self):
        worker = get_worker_info()
        paths = self.shards if worker is None else self.shards[worker.id::worker.num_workers]
        if not paths:
            return
        rng = random.Random(self.seed + (worker.id if worker else 0))
        paths = list(paths)
        if self.train:
            rng.shuffle(paths)
        # Worker shard assignment is explicit above; do not split it a second time.
        ds = wds.WebDataset(paths, shardshuffle=False, workersplitter=None).decode('pil')
        if self.train:
            ds = ds.shuffle(256, rng=rng)
        queues = {}
        for sample in ds:
            image = sample['bw.png'].convert('L')
            width, height = image.size
            if height != 64 or width > self.max_width or width < 16:
                raise ValueError(f"{sample['__key__']}: expected height 64 and width <= {self.max_width}, got {image.size}")
            text = sample['json']['text']
            if not text or len(self.alphabet.encode(text)) + 2 > 512:
                raise ValueError(f"Invalid or overlength label: {sample['__key__']}")
            # Character encoding validates coverage before training.
            self.alphabet.encode(text)
            bucket = next(b for b in (256, 512, 1024, 1600, 2048, 3072, 4096, 6144, 8192) if width <= b)
            queue = queues.setdefault(bucket, [])
            queue.append((1 - pil_to_tensor(image).float() / 255, text, sample['__key__']))
            capacity = max(1, min(self.batch_size, self.pixel_budget // bucket))
            if len(queue) >= capacity:
                yield collate(queue, self.alphabet)
                queue.clear()
        for queue in queues.values():
            if queue:
                yield collate(queue, self.alphabet)


def loader(paths, alphabet, args, train=False, epoch=0):
    dataset = Lines(paths, alphabet, train, args.seed + epoch, args.batch_size,
                    args.pixel_budget, args.max_width)
    return DataLoader(dataset, batch_size=None, num_workers=args.workers if train else 0,
                      pin_memory=True, generator=torch.Generator().manual_seed(args.seed + epoch))


def width_group(width):
    return 0 if width <= 1024 else 1 if width <= 2048 else 2 if width <= 4096 else 3


def probe_quotas(size):
    return [size // 4 + (i < size % 4) for i in range(4)]


def prepare_probe(paths, output, size):
    """Cache a deterministic width-balanced subset of validation, never test."""
    directory = output/'periodic_eval'
    directory.mkdir(exist_ok=True)
    manifest = directory/'manifest.json'
    signature = {'shards': [Path(p).name for p in paths], 'requested_samples': size,
                 'version': 1}
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['signature'] != signature:
            raise ValueError('Periodic subset configuration changed; use the original --periodic_eval_samples')
        if not all((directory/row['image']).exists() for row in saved['samples']):
            raise ValueError('Periodic subset images are missing')
        return directory, saved['samples']
    quotas = probe_quotas(size)
    counts, selected = [0]*4, []
    print(f'Preparing fixed validation subset: target counts by width {quotas}', flush=True)
    for sample in wds.WebDataset(paths, shardshuffle=False, workersplitter=None).decode('pil'):
        image = sample['bw.png'].convert('L')
        width, height = image.size
        group = width_group(width)
        if counts[group] >= quotas[group]:
            continue
        name = f'{len(selected):04d}.png'
        image.save(directory/name)
        selected.append({'image': name, 'key': sample['__key__'],
                         'text': sample['json']['text'], 'width': width, 'height': height})
        counts[group] += 1
        if counts == quotas:
            break
    if counts != quotas:
        warnings.warn(f'Validation split could supply only {counts}, requested {quotas}')
    manifest.write_text(json.dumps({'signature': signature, 'samples': selected}, ensure_ascii=False, indent=2))
    print(f'Fixed subset ready: {len(selected)} samples; width counts {counts}', flush=True)
    return directory, selected


def probe_batches(probe, alphabet, args):
    from PIL import Image
    directory, records = probe
    queues = {}
    for record in records:
        width = record['width']
        if record['height'] != 64 or not 16 <= width <= args.max_width:
            raise ValueError(f"Invalid probe image: {record['key']}")
        if not record['text'] or len(alphabet.encode(record['text'])) + 2 > 512:
            raise ValueError(f"Invalid probe label: {record['key']}")
        bucket = next(b for b in (256,512,1024,1600,2048,3072,4096,6144,8192) if width <= b)
        with Image.open(directory/record['image']) as image:
            pixels = 1 - pil_to_tensor(image.convert('L')).float()/255
        queue = queues.setdefault(bucket, [])
        queue.append((pixels, record['text'], record['key']))
        if len(queue) >= max(1, min(args.batch_size, args.pixel_budget // bucket)):
            yield collate(queue, alphabet)
            queue.clear()
    for queue in queues.values():
        if queue:
            yield collate(queue, alphabet)


def autocast(args):
    if args.precision == 'no':
        return contextlib.nullcontext()
    return torch.autocast('cuda', dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16)


def loss_for(model, batch, args):
    labels = batch['labels'].cuda(non_blocking=True)
    images = batch['images'].cuda(non_blocking=True)
    widths = batch['widths'].cuda(non_blocking=True)
    # Preserve train_mixed preprocessing: invert pixels, transpose, flip original vertical axis.
    images = images.transpose(-2, -1).flip(-1).contiguous()
    logits = model(images, labels[:, :-1], widths)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                           ignore_index=model.config.pad, label_smoothing=args.label_smoothing)
    return loss, images, widths


@torch.no_grad()
def evaluate(model, paths, alphabet, args, limit, output=None, batches=None):
    model.eval()
    errors = reference_chars = word_errors = reference_words = exact = count = capped = 0
    substitutions = deletions = insertions = 0
    loss_sum = 0.0
    rows, by_width = [], {}
    stream = open(output, 'w', encoding='utf-8') if output else None
    try:
        for batch in (loader(paths, alphabet, args) if batches is None else batches):
            if limit and count >= limit:
                break
            if limit and count + len(batch['texts']) > limit:
                n = limit - count
                batch = {k: v[:n] for k, v in batch.items()}
            with autocast(args):
                loss, images, widths = loss_for(model, batch, args)
                tokens = model.recognize(images, widths)
            predictions = alphabet.decode(tokens, [alphabet.eos])
            loss_sum += loss.item() * len(predictions)
            for i, (prediction, reference) in enumerate(zip(predictions, batch['texts'])):
                char = jiwer.process_characters(reference, prediction)
                word = jiwer.process_words(reference, prediction)
                e = char.substitutions + char.deletions + char.insertions
                r = char.hits + char.substitutions + char.deletions
                errors += e
                substitutions += char.substitutions
                deletions += char.deletions
                insertions += char.insertions
                reference_chars += r
                word_errors += word.substitutions + word.deletions + word.insertions
                reference_words += word.hits + word.substitutions + word.deletions
                exact += prediction == reference
                capped += not tokens[i].eq(alphabet.eos).any().item()
                width = int(batch['widths'][i])
                bucket = '<=1024' if width <= 1024 else '<=2048' if width <= 2048 else '<=4096' if width <= 4096 else '>4096'
                entry = by_width.setdefault(bucket, [0, 0, 0])
                entry[0] += e; entry[1] += r; entry[2] += 1
                record = {'key': batch['keys'][i], 'width': width, 'reference': reference, 'prediction': prediction}
                if stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                if len(rows) < 12 or width > min(row[1] for row in rows):
                    rows.append([wandb.Image(1 - images[i, :, :width, :].flip(-1).transpose(-2, -1).float().cpu()), width, reference, prediction])
                    rows = sorted(rows, key=lambda row: row[1], reverse=True)[:12]
                count += 1
    finally:
        if stream:
            stream.close()
    if not count:
        raise ValueError('Evaluation split is empty')
    metrics = {'cer': errors / reference_chars, 'wer': word_errors / reference_words,
               'exact_match': exact/count, 'loss': loss_sum/count, 'samples': count,
               'missing_eos_rate': capped/count,
               'substitution_rate': substitutions/reference_chars,
               'deletion_rate': deletions/reference_chars,
               'insertion_rate': insertions/reference_chars}
    for bucket, (e, r, n) in by_width.items():
        metrics[f'cer_width/{bucket}'] = e/r
        metrics[f'samples_width/{bucket}'] = n
    return metrics, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--output_dir', default='results_uhwr_long')
    p.add_argument('--mode', choices=['train', 'test'], default='train')
    p.add_argument('--checkpoint', help='LongLineModel folder; required for test')
    p.add_argument('--init_checkpoint', default='/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/best_model_mixed.pt', help='Original best_model_mixed.pt (fresh run only)')
    p.add_argument('--tokenizer_dir', default=str(Path(__file__).parent/'vocabs/ved'))
    p.add_argument('--resume', action='store_true', help='Resume output_dir/last.pt, replaying and skipping consumed batches if mid-epoch')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--pixel_budget', type=int, default=8192, help='Maximum sum of bucket widths per batch')
    p.add_argument('--max_width', type=int, default=8192)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--precision', choices=['no', 'fp16', 'bf16'], default='fp16')
    p.add_argument('--seed', type=int, default=24)
    p.add_argument('--eval_samples', type=int, default=2000, help='0 evaluates the entire validation split')
    p.add_argument('--eval_every_steps', type=int, default=500, help='Fixed-subset evaluation interval; 0 disables')
    p.add_argument('--periodic_eval_samples', type=int, default=128, help='Width-balanced fixed validation subset size')
    p.add_argument('--save_every_steps', type=int, default=500, help='Mid-epoch resume checkpoint interval; 0 disables')
    p.add_argument('--max_batches', type=int, default=0, help='Smoke-test batch limit per epoch; 0 is unlimited')
    p.add_argument('--wandb_project', default='uhwr-urdu-long-lines')
    p.add_argument('--wandb_name', default=None)
    p.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online')
    args = p.parse_args()
    if args.accumulation < 1 or not 16 <= args.max_width <= 8192 or min(args.batch_size, args.pixel_budget, args.epochs) < 1 or args.workers < 0 or args.eval_samples < 0 or args.max_batches < 0:
        p.error('Invalid batch/epoch/worker/limit settings; max_width must be between 16 and 8192')
    if args.resume and (args.mode != 'train' or args.checkpoint):
        p.error('--resume is only for training and cannot accompany --checkpoint')
    if min(args.eval_every_steps, args.save_every_steps) < 0 or args.periodic_eval_samples < 4:
        p.error('Step intervals must be nonnegative; periodic_eval_samples must be at least 4')
    if not torch.cuda.is_available():
        raise SystemExit('This training/evaluation script requires CUDA.')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        p.error('GPU does not support bf16; use fp16')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'train' and not args.resume and (output/'last.pt').exists():
        p.error('This output directory already has a run; use --resume or a new --output_dir')
    split_path = output / 'splits.json'
    if split_path.exists():
        split = json.loads(split_path.read_text())
    else:
        if args.mode == 'test' or args.resume:
            p.error('Existing splits.json is required for test/resume')
        files = sorted(Path(args.data_dir).resolve().glob('shard-*.tar'))
        if len(files) != 245:
            p.error(f'Expected 245 shards, found {len(files)}')
        random.Random(args.seed).shuffle(files)
        split = {k: [f.name for f in v] for k, v in
                 [('train', files[:220]), ('val', files[220:232]), ('test', files[232:])]}
        split_path.write_text(json.dumps(split, indent=2))
    paths = {k: [str(Path(args.data_dir).resolve()/name) for name in names] for k, names in split.items()}
    for names in paths.values():
        for name in names:
            if not Path(name).is_file():
                p.error(f'Missing shard: {name}')
    alphabet = Tokens(args.tokenizer_dir)
    vocab = alphabet.tokenizer.get_vocab()
    vocab_path = output/'vocabulary.json'
    if vocab_path.exists() and json.loads(vocab_path.read_text()) != vocab:
        p.error('Tokenizer differs from the saved run vocabulary')
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    if args.checkpoint:
        checkpoint_vocab = Path(args.checkpoint)/'vocabulary.json'
        if not checkpoint_vocab.exists() or json.loads(checkpoint_vocab.read_text()) != vocab:
            p.error('Checkpoint vocabulary file is missing or differs from the current tokenizer')
    if args.mode == 'test' and not args.checkpoint:
        p.error('--checkpoint is required in test mode')
    model = LongLineModel.from_pretrained(args.checkpoint) if args.checkpoint else LongLineModel(len(alphabet), alphabet.sos, alphabet.eos, alphabet.pad)
    if model.config.vocab_size != len(alphabet) or (model.config.bos, model.config.eos, model.config.pad) != (alphabet.sos, alphabet.eos, alphabet.pad):
        p.error('Checkpoint tokenizer configuration mismatch')
    config = model.config
    if args.init_checkpoint and not args.resume and not args.checkpoint and args.mode == 'train':
        model.initialize(args.init_checkpoint)
    elif args.mode == 'train' and not args.resume and not args.checkpoint:
        print('Training from scratch: no --init_checkpoint supplied', flush=True)
    model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    start, step, best = 0, 0, float('inf')
    resume_batches, best_periodic = 0, float('inf')
    resumed_total, resumed_seen = 0.0, 0
    run_id = None
    if args.resume:
        state = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if state['config'] != config.to_dict() or state['vocabulary'] != vocab:
            p.error('Resume configuration/vocabulary mismatch')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start, step, best = state['epoch'], state['step'], state['best']
        resume_batches = state.get('batches_seen', 0)
        best_periodic = state.get('best_periodic', float('inf'))
        resumed_total, resumed_seen = state.get('epoch_loss_sum', 0.0), state.get('epoch_microbatches', 0)
        if resume_batches and state.get('replay_settings') != {key: getattr(args,key) for key in ('seed','workers','batch_size','pixel_budget','max_width','accumulation','precision','label_smoothing','max_batches')}:
            p.error('Mid-epoch resume requires the original data/batch/precision settings')
        run_id = state['wandb_id']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    run = wandb.init(project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode,
                     config=vars(args), id=run_id, resume='allow' if run_id else None)
    run.config.update({'model': config.to_dict(), 'split_shards': {k: len(v) for k, v in split.items()}, 'loss': 'cross_entropy_only'})
    try:
        if args.mode == 'test':
            metrics, rows = evaluate(model, paths['test'], alphabet, args, 0, output/'test_predictions.jsonl')
            (output/'test_metrics.json').write_text(json.dumps(metrics, indent=2))
            run.log({**{f'test/{k}': v for k, v in metrics.items()}, 'test/examples': wandb.Table(columns=['image','width','reference','prediction'], data=rows)})
            print(json.dumps(metrics, indent=2)); return
        probe = prepare_probe(paths['val'], output, args.periodic_eval_samples) if args.eval_every_steps else None

        def save_state(epoch_index, batches_seen, loss_sum=0.0, microbatches=0):
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'epoch': epoch_index, 'batches_seen': batches_seen, 'step': step,
                     'best': best, 'best_periodic': best_periodic, 'wandb_id': run.id,
                     'config': config.to_dict(), 'vocabulary': vocab,
                     'epoch_loss_sum': loss_sum, 'epoch_microbatches': microbatches,
                     'replay_settings': {key: getattr(args,key) for key in ('seed','workers','batch_size','pixel_budget','max_width','accumulation','precision','label_smoothing','max_batches')},
                     'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
            temporary = output/'last.pt.tmp'
            torch.save(state, temporary)
            temporary.replace(output/'last.pt')

        def periodic_evaluation(epoch):
            nonlocal best_periodic
            begun = time.monotonic()
            metrics, rows = evaluate(model, None, alphabet, args, 0,
                                     output/'periodic_predictions.jsonl', probe_batches(probe, alphabet, args))
            run.log({**{f'periodic_val/{k}': v for k,v in metrics.items()},
                     'periodic_val/seconds': time.monotonic()-begun,
                     'epoch': epoch+1, 'optimizer_step': step,
                     'periodic_val/examples': wandb.Table(columns=['image','width','reference','prediction'], data=rows)})
            print(f'Periodic validation at optimizer step {step}: {metrics}', flush=True)
            if metrics['cer'] < best_periodic:
                best_periodic = metrics['cer']
                model.save_pretrained(output/'best_periodic')
                (output/'best_periodic'/'vocabulary.json').write_text(json.dumps(vocab, ensure_ascii=False))
            model.train()

        if probe is not None:
            periodic_evaluation(start)  # Establish inference CER before the next updates.
        for epoch in range(start, args.epochs):
            model.train(); optimizer.zero_grad(set_to_none=True)
            pending = 0
            total, seen = (resumed_total, resumed_seen) if epoch == start else (0.0, 0)
            batches_seen = 0
            epoch_started = time.monotonic()
            def update():
                nonlocal step, pending
                scaler.unscale_(optimizer)
                # Average over the actual number of microbatches, including the tail.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                step += 1; pending = 0
                return float(norm)
            for batch_idx, batch in enumerate(loader(paths['train'], alphabet, args, True, epoch)):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                if epoch == start and batch_idx < resume_batches:
                    continue
                batches_seen = batch_idx+1
                with autocast(args):
                    loss, _, _ = loss_for(model, batch, args)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Non-finite loss at epoch {epoch}, batch {batch_idx}')
                scaler.scale(loss).backward(); pending += 1
                total += loss.item(); seen += 1
                if pending == args.accumulation:
                    norm = update()
                    if step % 10 == 0:
                        log = {'train/loss': total/seen, 'train/lr': optimizer.param_groups[0]['lr'],
                               'train/grad_norm': norm, 'train/padded_width': batch['images'].size(-1),
                               'train/batch_size': len(batch['texts']), 'train/peak_vram_gb': torch.cuda.max_memory_allocated()/2**30,
                               'epoch': epoch+1, 'optimizer_step': step}
                        run.log(log); print(log, flush=True)
                    if args.save_every_steps and step % args.save_every_steps == 0:
                        save_state(epoch, batches_seen, total, seen)
                        print(f'Saved resumable checkpoint at optimizer step {step}', flush=True)
                    if probe is not None and step % args.eval_every_steps == 0:
                        periodic_evaluation(epoch)
                        if args.save_every_steps and step % args.save_every_steps == 0:
                            save_state(epoch, batches_seen, total, seen)
                if args.max_batches and batch_idx+1 >= args.max_batches:
                    break
            if pending:
                update()
            if not seen:
                raise RuntimeError('No training samples')
            run.log({'train/epoch_loss': total/seen, 'train/microbatches': seen,
                     'train/epoch_seconds_this_process': time.monotonic()-epoch_started,
                     'epoch': epoch+1, 'optimizer_step': step})
            metrics, rows = evaluate(model, paths['val'], alphabet, args, args.eval_samples)
            scheduler.step(metrics['cer'])
            run.log({**{f'val/{k}': v for k, v in metrics.items()}, 'epoch': epoch+1,
                     'optimizer_step': step, 'val/examples': wandb.Table(columns=['image','width','reference','prediction'], data=rows)})
            print(f'Epoch {epoch+1}: {metrics}', flush=True)
            if metrics['cer'] < best:
                best = metrics['cer']; model.save_pretrained(output/'best')
                (output/'best'/'vocabulary.json').write_text(json.dumps(vocab, ensure_ascii=False))
            model.save_pretrained(output/'latest')
            (output/'latest'/'vocabulary.json').write_text(json.dumps(vocab, ensure_ascii=False))
            save_state(epoch+1, 0)
    finally:
        run.finish()


if __name__ == '__main__':
    main()
