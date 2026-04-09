import torch
import math
import datasets
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm


tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
if tokenizer.mask_token is None:
    tokenizer.mask_token = "[MASK]"

model = AutoModelForMaskedLM.from_pretrained(config["model_name"])
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {total_params/1e6:.1f}M parameters")

# dataset preparation for Tinystories
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
if config["max_train_samples"]:
    dataset = dataset.select(range(config["max_train_samples"]))

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=config["seq_len"],
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )

tok_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tok_dataset = tok_dataset.with_format("torch")

dataloader = torch.utils.data.DataLoader(
    tok_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    drop_last=True,
)



optimizer= torch.optim.AdamW( model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"],)
total_steps = config["num_epochs"] * len(dataloader) // config["gradient_accumulation_steps"]
warmup_steps = int(config["warmup_ratio"] * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

accelerator = Accelerator(
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    mixed_precision=config["mixed_precision"],
)

model, optimizer, dataloader, scheduler = accelerator.prepare(
    model, optimizer, dataloader, scheduler
)


# Training loop (Algorithm 1)
model.train()
global_step = 0

for epoch in range(config["num_epochs"]):
    loss_cumsum = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Sample random t from Uniform(0,1) for each sample in the batch
        t = torch.rand(batch_size, 1, device=accelerator.device)
        t = t.clamp_min(1e-4)  
        t = t.expand(batch_size, config["seq_len"])

        # Masking each token independently with probability t
        mask = torch.bernoulli(t).bool()
        corrupted = input_ids.masked_fill(mask, tokenizer.mask_token_id)

        
        labels = input_ids.masked_fill(~mask, -100)

        with accelerator.accumulate(model):
            outputs = model(corrupted)
            logits = outputs.logits

            
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(batch_size, config["seq_len"])

            # Weight by 1/t (Equation 3 from LLaDA paper)
            loss = (per_token_loss / t).mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        
        if accelerator.sync_gradients:
            global_step += 1
            loss_cumsum += accelerator.gather(loss.detach()).mean().item()
            if (step + 1) % config["log_steps"] == 0:
                avg_loss = loss_cumsum / config["log_steps"]
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                loss_cumsum = 0


accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)


