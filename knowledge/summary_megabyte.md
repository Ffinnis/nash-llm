# MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

arXiv: https://arxiv.org/abs/2305.07185

## Useful architecture points

- MEGABYTE is a byte-level decoder architecture with three parts: patch embedder, global Transformer over patches, and local Transformer over bytes within each patch.
- Fixed-size byte patches reduce the expensive global sequence length by the patch size.
- The global model receives a patch-sized padding prefix and previous patches, avoiding future-byte leakage.
- The local model receives shifted bytes inside the target patch plus a per-byte projection of the global patch state, then predicts bytes autoregressively.
- The paper explicitly shows pseudocode: `bytes_global = pad + bytes[:-patch_size]`, `bytes_local = pad + bytes_input[:, :-1]`, global output reshaped to per-byte local conditioning, then a local Transformer produces logits.

## Implication for this repo

The naive patch head we tried was too weak because it predicted several bytes from the same patch latent without a real local causal Transformer. The repo implementation should follow the MEGABYTE shape:

1. Reconstruct the byte sequence for the training window.
2. Build global patch input from a learned global pad plus previous patches.
3. Run the existing large Transformer on patch latents.
4. Project each global patch state into per-byte local context.
5. Run a smaller causal local Transformer over the bytes in each patch with shifted local inputs.
6. Compute byte-level cross entropy and report bits-per-byte.

This keeps fixed-size patching for now. MEGABYTE also discusses cross-patch local attention, CNN patch encoders, and strided inference, but those are second-order follow-ups after the core local/global split works.
