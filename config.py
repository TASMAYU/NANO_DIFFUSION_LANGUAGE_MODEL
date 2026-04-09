config = {
    "model_name": "answerdotai/ModernBERT-base",
    "num_epochs": 3,
    "batch_size": 4,
    "seq_len": 256,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "log_steps": 20,
    "max_train_samples": 50000,
    "mixed_precision": "fp16",
}
