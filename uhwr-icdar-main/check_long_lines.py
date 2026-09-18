"""Model/data checks on the training machine; does not launch dataset training."""
import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from long_line_model import LongLineModel
from train_long_lines import Tokens, collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init_checkpoint', default='/home/ayeshaamjad/Desktop/PtHW_gen/uhwr-icdar_M/best_model_mixed.pt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    tokens = Tokens(Path(__file__).parent/'vocabs/ved')
    print(f'Tokenizer size: {len(tokens)}; BOS={tokens.sos}, EOS={tokens.eos}, PAD={tokens.pad}', flush=True)
    text = 'یہ مسجد اپنی طرزِ تعمیر کے حوالے سے منفرد ہے'
    assert tokens.decode(tokens.encode(text)[None])[0] == text
    model = LongLineModel(len(tokens), tokens.sos, tokens.eos, tokens.pad)
    original = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
    model.initialize(args.init_checkpoint)
    assert torch.equal(model.transformer_decoder.transformer.wpe.weight, original['transformer_decoder.transformer.wpe.weight'])
    old = original['transformer_encoder.pos_embedding.pos']
    assert torch.equal(model.transformer_encoder.pos_embedding.pos[:, :old.size(1)], old)
    model.to(args.device).eval()
    for width in (256, 7188, 8192):
        batch = collate([(1-pil_to_tensor(Image.new('L', (width,64), 255)).float()/255, text, 'check')], tokens)
        assert int(batch['widths'][0]) == width
        images = batch['images'].to(args.device).transpose(-2,-1).flip(-1).contiguous()
        widths = batch['widths'].to(args.device)
        labels = batch['labels'].to(args.device)
        with torch.no_grad():
            features, mask = model.encode(images, widths)
            assert features.size(1) == math.ceil(width/8)*8//4-3
            assert int(mask.sum()) == width//4-3
            logits = model(images, labels[:, :-1], widths)
            assert logits.shape == (1, labels.size(1)-1, len(tokens))
            assert torch.isfinite(logits).all()
            generated = model.transformer_decoder.generate(
                input_ids=labels[:, :1], attention_mask=torch.ones_like(labels[:, :1]),
                encoder_hidden_states=features, encoder_attention_mask=mask,
                max_new_tokens=4, do_sample=False, use_cache=True,
                eos_token_id=tokens.eos, pad_token_id=tokens.pad,
            )
            assert generated.size(1) <= 5
        print(f'PASS width={width}, visual_tokens={features.size(1)}, cached generation', flush=True)
    # Unequal widths must mask the padded part of the short sample.
    mixed = collate([(torch.zeros(1,64,w), text, str(w)) for w in (256,512)], tokens)
    images = mixed['images'].to(args.device).transpose(-2,-1).flip(-1).contiguous()
    with torch.no_grad():
        _, mask = model.encode(images, mixed['widths'].to(args.device))
    assert mask.sum(1).tolist() == [61,125]
    model.train()
    labels = mixed['labels'].to(args.device)
    logits = model(images, labels[:, :-1], mixed['widths'].to(args.device))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1,len(tokens)), labels[:,1:].reshape(-1), ignore_index=tokens.pad)
    loss.backward()
    assert model.cnn_encoder.conv1[0].weight.grad is not None
    assert torch.isfinite(model.cnn_encoder.conv1[0].weight.grad).all()
    assert model.ctc_head.proj.weight.grad is None
    print('PASS unequal-width masks and finite gradients through the CNN; CTC disabled')
    print('PASS checkpoint compatibility, positional extension, tokenizer round trip, and full-width forward checks')


if __name__ == '__main__':
    main()
