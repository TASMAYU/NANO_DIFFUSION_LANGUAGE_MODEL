# Nano Diffusion Language Model
![Diffusion LLM INFERENCE](p2.gif)
A small implementation of a Bonsai diffusion language models based on LLaDA paper.

I have used ModernBERT (bidirectional transformer) as a "mask predictor" trained with:
- Random masking ratio t ~ U[0,1]
- 1/t loss weighting added in the loss
- Iterative remasking sampling

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python diffusion_train.py

# Generate
python inference.py
