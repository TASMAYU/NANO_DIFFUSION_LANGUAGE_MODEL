@torch.no_grad()
def generate_text(prompt=None, num_steps=64, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        seq_len = config["seq_len"]
        input_ids = torch.full((1, seq_len), tokenizer.mask_token_id, device=device)
        input_ids[0, :prompt_len] = torch.tensor(prompt_ids, device=device)
        prompt_mask = torch.zeros(1, seq_len, device=device)
        prompt_mask[0, :prompt_len] = 1
    else:
        input_ids = torch.full((1, config["seq_len"]), tokenizer.mask_token_id, device=device)
        prompt_mask = torch.zeros(1, config["seq_len"], device=device)

    for step in range(num_steps):
        t = 1.0 - (step / num_steps)       # current timestep
        s = 1.0 - ((step + 1) / num_steps) # next timestep

        outputs = model(input_ids)
        logits = outputs.logits

        # Only consider positions that are masked AND not part of prompt
        mask_positions = (input_ids == tokenizer.mask_token_id) & (prompt_mask == 0)

        
        predictions = logits.argmax(dim=-1)

        # Remask probability = s/t
        remask_prob = s / t
        remask = torch.rand_like(input_ids.float()) < remask_prob

        new_input_ids = input_ids.clone()
        new_input_ids[mask_positions] = predictions[mask_positions]
        new_input_ids[remask & mask_positions] = tokenizer.mask_token_id
        input_ids = new_input_ids

    # Decoding step here removing mask tokens and any tokens after EOS if needed
    output_ids = input_ids[0].cpu().tolist()
    output_ids = [id for id in output_ids if id != tokenizer.mask_token_id]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text
