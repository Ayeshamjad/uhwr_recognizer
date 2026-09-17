import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from tokeniser import get_tokenizer


CHECKPOINT_DIR = Path(__file__).parent / \
    "decoder_pretrain_tokenizer_bos_eos" / "checkpoint-32452"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick smoke-test for the GPT-2 decoder using an Urdu prompt.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=" ٹستنگ کے لیے اردو متن جو آپ ڈیکوڈر میں فیڈ کرنا چاہتے ہیں اور پاکستان",
        help="Urdu text you want to feed into the decoder.",
    )
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum length for generation.")
    parser.add_argument("--num_beams", type=int, default=4,
                        help="Beam width for generation.")
    return parser.parse_args()


def load_decoder(device):
    if not CHECKPOINT_DIR.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_DIR.resolve()}")

    tokenizer = get_tokenizer()
    tokenizer.padding_side = "right"

    decoder = GPT2LMHeadModel.from_pretrained(CHECKPOINT_DIR)
    decoder.resize_token_embeddings(len(tokenizer))

    # Align generation config with the settings we rely on elsewhere.
    decoder.config.decoder_start_token_id = tokenizer.bos_token_id
    decoder.config.pad_token_id = tokenizer.pad_token_id
    decoder.config.eos_token_id = tokenizer.eos_token_id
    decoder.config.max_length = 256
    decoder.config.early_stopping = False
    decoder.config.no_repeat_ngram_size = 0
    decoder.config.length_penalty = 1.0
    decoder.config.num_beams = 4
    decoder.config.temperature = 1.0

    decoder.to(device)
    decoder.eval()
    return decoder, tokenizer


def run_generation(decoder, tokenizer, prompt, max_length, num_beams):
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(next(decoder.parameters()).device)
               for key, value in encoded.items()}

    with torch.no_grad():
        generated_ids = decoder.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=decoder.config.early_stopping,
            no_repeat_ngram_size=decoder.config.no_repeat_ngram_size,
            length_penalty=decoder.config.length_penalty,
            temperature=decoder.config.temperature,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder, tokenizer = load_decoder(device)

    generated = run_generation(decoder, tokenizer, prompt=args.prompt,
                               max_length=args.max_length, num_beams=args.num_beams)

    print("==== Prompt ====")
    RTL_EMBED = "\u202B"
    POP_DIRECTIONAL = "\u202C"

    print(f"{RTL_EMBED}{args.prompt}{POP_DIRECTIONAL}")
    print("\n==== Decoder Output ====")
    print(f"{RTL_EMBED}{generated}{POP_DIRECTIONAL}")


if __name__ == "__main__":
    main()
