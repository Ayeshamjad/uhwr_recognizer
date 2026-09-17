import torch

def greedy_decode(model, pixel_values, max_len, tokenizer):
    """
    Greedy decoding for inference.
    """
    model.eval()
    with torch.no_grad():
        cnn_output = model.cnn_encoder(pixel_values)
        projected_output = model.projection(cnn_output)
        encoder_output = model.transformer_encoder(projected_output)

        # Start with BOS token
        tgt = torch.full((pixel_values.size(0), 1), tokenizer.bos_token_id, dtype=torch.long).to(pixel_values.device)

        for _ in range(max_len):
            decoder_output = model.transformer_decoder(input_ids=tgt, encoder_hidden_states=encoder_output)
            next_token_logits = decoder_output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            tgt = torch.cat([tgt, next_token], dim=1)
    return tgt

def beam_search_decode(model, pixel_values, tokenizer):
    """
    Beam search decoding for inference.
    This is a simplified implementation. For a robust implementation, consider using libraries like `transformers.generation_utils`.
    """
    model.eval()
    with torch.no_grad():
        cnn_output = model.cnn_encoder(pixel_values)
        projected_output = model.projection(cnn_output)
        encoder_output = model.transformer_encoder(projected_output)

        transformer = model.transformer_decoder

        transformer.config.decoder_start_token_id = tokenizer.bos_token_id
        transformer.config.pad_token_id = tokenizer.pad_token_id
        decoder_cfg = getattr(transformer.config, "decoder", None)
        if decoder_cfg is not None and hasattr(decoder_cfg, "vocab_size"):
            transformer.config.vocab_size = decoder_cfg.vocab_size

        transformer.config.eos_token_id = tokenizer.eos_token_id
        transformer.config.max_length = 256
        transformer.config.early_stopping = False
        transformer.config.no_repeat_ngram_size = 0
        transformer.config.length_penalty = 1
        transformer.config.num_beams = 4
        transformer.config.temperature = 1

        outputs = transformer.generate(
            input_ids=None,
            encoder_hidden_states=encoder_output,
            max_length=transformer.config.max_length,
            num_beams=transformer.config.num_beams,
            early_stopping=transformer.config.early_stopping,
            no_repeat_ngram_size=transformer.config.no_repeat_ngram_size,
            length_penalty=transformer.config.length_penalty,
            temperature=transformer.config.temperature,
            bos_token_id=transformer.config.decoder_start_token_id,
            eos_token_id=transformer.config.eos_token_id,
            pad_token_id=transformer.config.pad_token_id,
        )
    return outputs
