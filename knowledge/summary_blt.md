# BLT: Byte Latent Transformer

arXiv: https://arxiv.org/abs/2412.09871

## Useful architecture points

- BLT is a tokenizer-free byte model that groups bytes into patches and runs the expensive latent Transformer over patch representations.
- Unlike fixed MEGABYTE patches, BLT emphasizes dynamic patching: entropy-based boundaries allocate more compute to difficult bytes and longer patches to predictable spans.
- The architecture has a lightweight local encoder, a large latent global Transformer, and a lightweight local decoder.
- BLT uses cross-attention between byte representations and patch representations in both encoder and decoder. The decoder cross-attention is reported as especially important.
- BLT uses byte n-gram hash embeddings to recover lexical/morphological information that fixed vocab tokenizers usually provide.
- The paper reports that static patching and space patching help efficiency but are not enough by themselves; entropy patching plus architectural improvements are what close the gap at larger scale.

## Implication for this repo

The practical sequence for `nash-llm` should be:

1. Implement the MEGABYTE fixed-patch core first, because it fits the current memmap dataset and rectangular batch tensors.
2. Add local decoder capacity before increasing patch size, since weak local decoding caused high early loss.
3. Consider space patching next because it is much simpler than entropy patching and BLT reports it as a strong simple baseline.
4. Add n-gram hash embeddings if fixed/space patches remain behind token models.
5. Treat true entropy patching as a later data-loader/model interface change, because dynamic patch counts require variable patch packing or padding.

The current patch implementation is therefore a MEGABYTE-style stepping stone, not full BLT.
