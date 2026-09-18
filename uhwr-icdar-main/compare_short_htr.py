"""Read-only checkpoint comparison on labeled words/short lines; no training."""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import jiwer
import numpy as np
import torch
import webdataset as wds
from PIL import Image

from long_line_model import LongLineModel
from model.joint_model import JointModel
from train_long_lines import Tokens


def original_pixels(image):
    """Exact no-augmentation MixedHWRDataset preprocessing/bucket assignment."""
    array = np.asarray(image.convert('L'), dtype=np.float32)/255
    width, height = image.size
    scaled_width = int(width*64/height)
    bucket = next((b for b in (256,512,1024,1600) if scaled_width <= b),1600)
    array = array.swapaxes(-2,-1)[...,::-1]
    target = np.ones((bucket,64),dtype=np.float32)
    scale = min(bucket/array.shape[0],64/array.shape[1])
    new_x, new_y = max(1,int(array.shape[0]*scale)), max(1,int(array.shape[1]*scale))
    target[:new_x,:new_y] = cv2.resize(array,(new_y,new_x))
    return torch.from_numpy(1-target)[None,None]


def modified_pixels(image):
    # Accept external examples of other heights, normalize uniformly to 64.
    # Shard images are already height 64 and receive no resizing.
    image = image.convert('L')
    if image.height != 64:
        image = image.resize((max(16,round(image.width*64/image.height)),64),Image.Resampling.LANCZOS)
    width = image.width
    if not 16 <= width <= 8192:
        raise ValueError(f'Unsupported normalized width {width}')
    array = 1-np.asarray(image,dtype=np.float32)/255
    padded = np.zeros((64,math.ceil(width/8)*8),dtype=np.float32)
    padded[:,:width] = array
    return torch.from_numpy(padded.T[:,::-1].copy())[None,None], torch.tensor([width])


def select_samples(args):
    samples = []
    if args.samples_csv:
        csv_path = Path(args.samples_csv)
        root = Path(args.image_root) if args.image_root else csv_path.parent
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            if not {'file_name','text'} <= set(reader.fieldnames or []):
                raise ValueError('CSV must contain file_name,text columns')
            for row in reader:
                with Image.open(root/row['file_name']) as image:
                    samples.append({'key':row['file_name'],'text':row['text'],'image':image.convert('L').copy()})
                if len(samples) >= args.count:
                    break
        return samples
    if not args.data_dir:
        raise ValueError('Supply --samples_csv or --data_dir')
    split=json.loads(Path(args.split_manifest).read_text())
    paths=[str(Path(args.data_dir)/name) for name in split['val']]
    counts={'word':0,'short_line':0}
    quotas={'word':args.count//2,'short_line':args.count-args.count//2}
    for sample in wds.WebDataset(paths,shardshuffle=False,workersplitter=None).decode('pil'):
        text=sample['json']['text']; image=sample['bw.png'].convert('L')
        width=image.width*64/image.height
        if not text.strip() or not 16 <= width <= 1024 or len(text) > 100:
            continue
        group='word' if len(text.split())==1 else 'short_line'
        if counts[group] >= quotas[group]:
            continue
        samples.append({'key':sample['__key__'],'text':text,'image':image.copy()})
        counts[group]+=1
        if counts==quotas:
            break
    print(f'Selected held-out rendered samples: {counts}',flush=True)
    return samples


@torch.no_grad()
def decode(model,images,widths,tokens,beams,original=False):
    if original:
        memory=model.transformer_encoder(model.projection(model.cnn_encoder(images)))
        source_mask=None
    else:
        memory,source_mask=model.encode(images,widths)
    prompt=torch.full((1,1),tokens.sos,device=images.device,dtype=torch.long)
    kwargs=dict(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                encoder_hidden_states=memory,max_new_tokens=511,num_beams=beams,
                do_sample=False,use_cache=True,bos_token_id=tokens.sos,
                eos_token_id=tokens.eos,pad_token_id=tokens.pad,
                early_stopping=beams>1,length_penalty=1.0,no_repeat_ngram_size=0)
    if source_mask is not None:
        kwargs['encoder_attention_mask']=source_mask
    # Original mixed beam evaluation did not suppress special tokens. Its greedy
    # control uses the same suppression as our modified pipeline for comparison.
    if not original or beams==1:
        kwargs['suppress_tokens']=[tokens.sos,tokens.pad]
    ids=model.transformer_decoder.generate(**kwargs)[:,1:]
    return tokens.decode(ids)[0],not ids.eq(tokens.eos).any().item()


def metrics(records):
    references=[r['reference'] for r in records]
    predictions=[r['prediction'] for r in records]
    char=jiwer.process_characters(references,predictions)
    word=jiwer.process_words(references,predictions)
    return {'samples':len(records),'cer':char.cer,'wer':word.wer,
            'exact_match':sum(p==r for p,r in zip(predictions,references))/len(records),
            'missing_eos_rate':sum(r['missing_eos'] for r in records)/len(records),
            'substitutions':char.substitutions,'deletions':char.deletions,'insertions':char.insertions}


def load_finetuned(path,tokens):
    path=Path(path)
    if path.is_dir():
        vocabulary=json.loads((path/'vocabulary.json').read_text())
        model=LongLineModel.from_pretrained(path)
    else:
        # Only use last.pt files made by our trainer; they include Python RNG state.
        state=torch.load(path,map_location='cpu',weights_only=False)
        vocabulary=state['vocabulary']
        config=dict(state['config']); config.pop('architecture')
        model=LongLineModel(**config); model.load_state_dict(state['model'],strict=True)
        model.comparison_checkpoint_info={'optimizer_step':state['step'],'epoch_index':state['epoch']}
    if vocabulary != tokens.tokenizer.get_vocab():
        raise ValueError('Fine-tuned checkpoint tokenizer differs')
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original_checkpoint',default='/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/best_model_mixed.pt')
    parser.add_argument('--finetuned_checkpoint',default='results_uhwr_long/last.pt')
    parser.add_argument('--samples_csv',help='Optional original evaluation examples: file_name,text CSV')
    parser.add_argument('--image_root',help='Root for relative CSV image filenames')
    parser.add_argument('--data_dir',help='Rendered TAR directory, used if no CSV is supplied')
    parser.add_argument('--split_manifest',default='results_uhwr_long/splits.json')
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--output_dir',default='short_htr_comparison')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.count < 2:
        parser.error('--count must be at least 2')
    tokens=Tokens(Path(__file__).parent/'vocabs/ved')
    samples=select_samples(args)
    if not samples:
        raise ValueError('No labeled examples selected')
    for sample in samples:
        if not sample['text'].strip() or len(tokens.encode(sample['text']))+2 > 512:
            raise ValueError(f"Invalid or overlength text: {sample['key']}")
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    selected=[]
    for index,sample in enumerate(samples):
        filename=f'{index:03d}.png';sample['image'].save(output/filename)
        selected.append({'file_name':filename,'text':sample['text'],'source_key':sample['key']})
    (output/'samples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
    with (output/'samples.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['file_name','text','source_key']);writer.writeheader();writer.writerows(selected)
    original=JointModel(256,8,3,1024,tokens.tokenizer,256,8,3,512,freeze_decoder=False,decoder_path=None)
    original.load_state_dict(torch.load(args.original_checkpoint,map_location='cpu',weights_only=True),strict=True)
    original.to(args.device).eval()
    modified=LongLineModel(len(tokens),tokens.sos,tokens.eos,tokens.pad)
    modified.initialize(args.original_checkpoint);modified.to(args.device).eval()
    finetuned=load_finetuned(args.finetuned_checkpoint,tokens).to(args.device).eval()
    metadata={**vars(args),'finetuned_snapshot':getattr(finetuned,'comparison_checkpoint_info',{}),
              'source':'external_labeled_images' if args.samples_csv else 'rendered_validation_shards',
              'note':'Small diagnostic; does not verify an original-dataset accuracy claim.'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    variants=[('original_beam4',original,True,4),('original_greedy_control',original,True,1),
              ('modified_original_greedy',modified,False,1),('modified_original_beam4',modified,False,4),
              ('finetuned_greedy',finetuned,False,1),('finetuned_beam4',finetuned,False,4)]
    records=[]
    for index,sample in enumerate(samples):
        print(f"Example {index+1}/{len(samples)}: {sample['key']} | {sample['text']}",flush=True)
        for name,model,is_original,beams in variants:
            if is_original:
                images=original_pixels(sample['image']);widths=None
            else:
                images,widths=modified_pixels(sample['image']);widths=widths.to(args.device)
            prediction,missing=decode(model,images.to(args.device),widths,tokens,beams,is_original)
            row={'variant':name,'key':sample['key'],'group':'word' if len(sample['text'].split())==1 else 'short_line',
                 'reference':sample['text'],'prediction':prediction,'cer':jiwer.cer(sample['text'],prediction),'missing_eos':missing}
            records.append(row)
            print(f'  {name}: CER={row["cer"]:.3f} | {prediction}',flush=True)
            with (output/'predictions.jsonl').open('a' if len(records)>1 else 'w',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={}
    for name,_,_,_ in variants:
        matching=[r for r in records if r['variant']==name]
        summary[name]=metrics(matching)
        summary[name]['by_group']={g:metrics([r for r in matching if r['group']==g]) for g in ('word','short_line') if any(r['group']==g for r in matching)}
    (output/'metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(f'Comparison saved to {output.resolve()}')


if __name__=='__main__':
    main()
