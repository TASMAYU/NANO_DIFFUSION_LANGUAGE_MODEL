# Nano Diffusion Language Model
![diffusion_unmasking-2](https://github.com/user-attachments/assets/fca1d48f-1445-4133-82e7-d0d5eb4d2b20)


A small implementation of a Bonsai(Nano) diffusion language models based on LLaDA paper.

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
