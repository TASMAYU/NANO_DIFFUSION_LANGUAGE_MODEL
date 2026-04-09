# Nano Diffusion Language Model

A small implementation of a Bonsai diffusion language models based on LLaDA paper.


This model uses a bidirectional transformer (ModernBERT) as a "mask predictor" trained with:
- Random masking ratio t ~ U[0,1]
- 1/t loss weighting 
- Iterative remasking sampling

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python diffusion_train.py

# Generate
python demo.py
