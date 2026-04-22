## Small-model pretraining comparators for nash-llm

Goal: assess whether the current `~124M` model's `val_loss` is obviously bad or broadly plausible.

### Closest useful arXiv comparators

1. Inheritune: Training Smaller Yet More Attentive Language Models
   - URL: https://arxiv.org/abs/2404.08634
   - Relevance: discusses GPT-2 family training on OpenWebText-9B and mentions FineWeb-Edu.
   - Useful point: smaller GPT-style models can match larger counterparts' validation loss under better training recipes. Good for optimization ideas, but not a direct apples-to-apples loss target for a 124M FineWeb-Edu run.

2. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
   - URL: https://arxiv.org/abs/2402.00159
   - Relevance: trains 1.2B models on several open corpora and compares intrinsic fit.
   - Useful point: on Paloma, mixed corpora like Dolma and Pile beat single-source web corpora like RefinedWeb. This suggests data mixture/quality may matter more than small optimizer tweaks if the objective is lower broad-domain loss.

3. OLMo: Accelerating the Science of Language Models
   - URL: https://arxiv.org/abs/2402.00838
   - Relevance: open 1B-scale baseline trained on Dolma with explicit pretraining settings.
   - Useful point: evaluates with Paloma bits-per-byte after decontamination, not raw held-out cross-entropy on the pretraining corpus. Good reference for protocol rigor, not for a directly comparable loss number.

4. OpenELM: An Efficient Language Model Family with Open Training and Inference Framework
   - URL: https://arxiv.org/abs/2404.14619
   - Relevance: open small-to-mid-size models trained on a public mixture including RefinedWeb, Pile, RedPajama, and Dolma.
   - Useful point: 1.1B OpenELM outperforms OLMo 1.2B on downstream evals while using fewer pretraining tokens, reinforcing that architecture/data/recipe differences can swamp raw parameter-count comparisons.

5. Cerebras-GPT: Open Compute-Optimal Language Models Trained on the Cerebras Wafer-Scale Cluster
   - URL: https://arxiv.org/abs/2304.03208
   - Relevance: includes a 111M model and explicitly follows Chinchilla-style compute-optimal scaling.
   - Useful point: for a ~100M-scale model, compute-optimal training implies substantially more than 1B tokens. This supports treating a 1B-token run as informative but likely undertrained rather than definitive.

6. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
   - URL: https://arxiv.org/abs/2304.01373
   - Relevance: public checkpoints for 160M-scale models.
   - Useful point: small public baseline suites are usually trained much longer than 1B tokens, so they are better for studying scaling trends than for setting a direct loss target for a short FineWeb-Edu run.

### Practical conclusion

- I did not find an arXiv source with a directly comparable published `val_loss` for `124M + FineWeb-Edu + 1B tokens + GPT-2 tokenizer`.
- Your current value should therefore be treated as an internal baseline, not something that can be cleanly labeled good or bad from the literature alone.
- The literature does support three likely conclusions:
  - `1B` tokens is probably still short for a `~124M` model if the goal is best achievable loss.
  - data mixture/quality is a high-leverage axis; single-source web data often underperforms mixed curated corpora on broad intrinsic evaluation.
  - comparisons across papers should use a standardized intrinsic benchmark like Paloma or the same exact held-out validation split, otherwise raw loss numbers are not trustworthy.
