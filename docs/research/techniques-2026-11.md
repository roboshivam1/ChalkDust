# Getting big-model quality out of small models for ChalkDust (techniques, November 2026)

Written 17 November 2026 by a research session for ChalkDust (branch `research/small-local-models`). It follows on from `docs/research/chalkdust-research-2026-10.md` (26 Oct 2026, partial copy).

How to read this:

- Claims about models, tools, licences, dates and numbers come from pages fetched in this session, listed under Sources at the end, unless they carry a tag.
- **[unverified]** means the claim could not be confirmed from a primary source. **[secondary]** and **[anecdotal]** mean it was seen only in search results or third-party pages. **[inference]** marks this doc's own reasoning.
- Text in quotation marks is copied from the source; everything else is paraphrase.
- Some pages showed only a day and month. Where the year is filled in from page context, it is marked **(year inferred)**.
- VRAM and data-size figures are **estimates** unless a source says they were measured.
- Nothing was run on the lab machine. Anything that needs it is listed under "Run when back".
- Base-model choice belongs to the models research. This doc uses "a 4B-class dense model with an Apache-2.0 licence" as the working assumption. Qwen3.5-4B and Gemma 4 E4B both qualify, per their Hugging Face cards. [checked 2026-11-17: both cards show `apache-2.0`. Note that the Qwen3.5-4B and 9B cards describe models with a vision encoder and a hybrid Gated DeltaNet / gated-attention layout, released February 2026 per their citation. Newer releases exist: Qwen3.6, so far only at 27B and 35B-A3B per Unsloth's Qwen3.6 page, so no small Qwen3.6 was found; and Gemma 4 additions, MTP heads (16 Apr 2026) and a "Gemma 4 12B Unified" model (3 Jun 2026), per Google's Gemma release log. No Gemma 5 was found]

## Outcome summary

1. **The biggest win needs no training.** At inference, use the validator for best-of-N plus a repair loop that feeds its typed refusal back into the model. Add grammar-constrained decoding with flattened per-component schemas. The evidence is consistent: small models gain the most from extra samples when an automatic verifier exists. Self-repair works poorly when the model critiques itself and much better with precise external feedback. ChalkDust's validator is that feedback, and it is deterministic.
2. **The first training step is rejection-sampled SFT with LoRA.** Generate with a teacher you are allowed to distil from, keep only the samples that pass the validator, and fine-tune a 4B-class model:
   - LoRA on all linear layers, rank 16-32, rsLoRA on.
   - Learning rate about 10x what full fine-tuning would use. rsLoRA changes the effective scale, so sweep it on a small pilot first.
   - 2-3 epochs.
   - This fits in 12 GB with QLoRA (estimate: 5-8 GB).
   - Fancier LoRA variants buy at most a few points and are not reliable. rsLoRA is the one worth switching on, because it costs nothing.
3. **The choice of teacher is a licence decision before it is a quality decision.**
   - Anthropic's and Google's API terms forbid using their services to train competing models. OpenAI's terms could not be fetched **[unverified]**. [corrected 2026-11-17: OpenAI's consumer Terms of Use (effective 1 Jan 2026) were fetched and forbid "Use Output to develop models that compete with OpenAI". The API Services Agreement still returned HTTP 403]
   - DeepSeek's API terms (effective 29 Apr 2026) explicitly allow using outputs for "training other models (such as model distillation)".
   - Qwen3.5 and Gemma 4 are plain Apache-2.0, which puts no restriction on outputs. Google's separate Gemma Terms of Use and Prohibited Use Policy cover earlier Gemma models, not Gemma 4.
4. **RL with the validator as reward is the second training step, for job 3 (component params) only.**
   - Use TRL's default DAPO loss, `beta=0`, and a hard validator gate plus a content floor so the model cannot win by emitting trivially small or RawScene-heavy specs.
   - It needs vLLM for speed, and vLLM needs **WSL2** on this laptop.
   - A 4B model is at the edge of 12 GB. Unsloth's own benchmark of Qwen3-4B GRPO on a 16 GB T4 reports about 13 GiB, usually 10-11 GB, with its Standby mode. A 1.5-2B model fits comfortably. [corrected 2026-11-17: the Unsloth page does not state the LoRA/QLoRA setting, sequence length or number of generations for that run, so it is a rough anchor, not a like-for-like figure for ChalkDust's 4-8k prompts]
   - [corrected 2026-11-17: zero-advantage group filtering is documented for TRL's `AsyncGRPOTrainer` (v1.17.0 notes), which needs a separate vLLM server. The `GRPOConfig` docs show no such option for the plain `GRPOTrainer`; they only log `frac_reward_zero_std`]
5. **On-policy distillation is the most compute-efficient way to close the remaining gap, but on one 12 GB GPU it is only practical in a reduced form.**
   - It needs teacher log-probs on the student's own tokens, so only an open-weight teacher with a compatible tokenizer works.
   - TRL's async version needs three separate GPUs.
   - The laptop-friendly variant is TRL's SDFT. The same model, shown a validator-passing reference spec (and optionally a refusal computed in advance for a typical bad spec) as `privileged_context`, acts as its own teacher. `privileged_context` is a fixed dataset column, so it cannot carry a live refusal of the student's current sample. This is an **untested idea** for ChalkDust, but it fits in 12 GB. [corrected 2026-11-17: "fits in 12 GB" is an estimate, not a measurement; SDFT lives in `trl.experimental.sdft`, so its API can change between releases. TRL v1.18.0 also added privileged-context distillation to `GOLDTrainer` (`use_privileged_context=True`), but the GOLD docs limit it to "text, off-policy, standard-JSD training" (`lmbda` must be 0), so it is not an on-policy alternative to SDFT]
6. **Inference speed on B and C:**
   - Ship 4-bit k-quants (Q4_K_M, built with an importance matrix from ChalkDust's own prompts), or a QAT checkpoint converted correctly.
   - Try draftless n-gram speculative decoding first, since JSON repeats itself.
   - EAGLE-3's 2-3x numbers come from a 48 GB desktop GPU. No laptop numbers exist, and a fine-tuned model would need a re-trained head. [corrected 2026-11-17: the 8B dense results span 1.62-3.28x across three prompts; the MoE rows (Qwen3-30B-A3B, gpt-oss-20B) were measured on a DGX Spark 128 GB, not the 48 GB card. See section 6]
   - NVFP4 helps only machine A (Blackwell); llama.cpp's Metal backend does not run it **[secondary source]**.
7. **Tooling on machine A:** native Windows is fine for SFT, QLoRA and llama.cpp inference. RL and on-policy distillation should run in WSL2 because vLLM is Linux-only. The current documented stack is PyTorch 2.15 (CUDA 13.2 default wheel), triton-windows `<3.10`, bitsandbytes with CUDA 13.0-13.2 Windows wheels that include sm_120, and TRL 1.18.1. Pin and test before trusting it. [corrected 2026-11-17: the PyTorch cu130 wheel index has no torch 2.15.0 Windows wheel (it stops at 2.14.1). 2.15.0 Windows wheels are on the cu132 and cu134 indexes; use **cu132**, because bitsandbytes' Windows CUDA wheels stop at 13.2. For WSL2, TRL 1.18 supports only `vllm>=0.23.0,<0.34.0`, and TRL's notes say vLLM 0.33.0 requires torch 2.14.0, but vLLM v0.34.0 was released on 16 Nov 2026 (GitHub API). Pin vLLM 0.33.x for TRL]

## 1. Distillation and SFT on teacher outputs with rejection sampling

### What the evidence says

- **Rejection-sampled SFT** is the standard cheap step: sample several outputs per prompt, keep the correct ones, fine-tune on them.
  - Yuan et al. (arXiv 2308.01825, Aug 2023) found it helps more when the kept data has more *distinct* reasoning paths, and helps weaker models more.
  - Pooling rejection samples from several models raised LLaMA-7B from 35.9% (plain SFT) to 49.3% on GSM8K.
  - For ChalkDust, the validator's deterministic rungs (pydantic schema, semantic checks, geometric probe) are the rejection filter.
- **On-policy distillation** is the strongest-documented step after SFT. Thinking Machines' "On-Policy Distillation" (Kevin Lu et al., 27 Oct 2025) samples from the *student* and has the teacher grade every token with per-token reverse KL.
  - On Qwen3-8B-Base, starting from a 400K-prompt SFT checkpoint at 60% on AIME'24, it reached 70% in about 150 steps (about 77K prompts, 4 samples each).
  - The authors estimate 9-30x less compute than pushing SFT to 2M examples.
  - They cite the Qwen3 report: RL took 17,920 GPU hours to reach 67.6% on AIME'24; on-policy distillation took 1,800 GPU hours to reach 74.4%.
  - It works with LoRA. At rank 32, LoRA trailed full fine-tuning by 13% after SFT but only by 6% after on-policy distillation.
  - The post calls the reverse-KL reward "unhackable", unlike a validator reward (section 3).
  - A June 2026 note on the page says their recipes now use Qwen3.5-9B-Base as the student and Qwen3.5-9B as the teacher.
- **Frontier labs use it routinely.** Hugging Face's "Distillation in 2026 (so far)" (8 Jul 2026) lists on-policy distillation in DeepSeek-V4, MiMo-V2-Flash, GLM-5 and Nemotron 3 Ultra, often with same-size specialist teachers. [corrected 2026-11-17: the post explicitly calls DeepSeek-V4 and MiMo-V2-Flash (MOPD) on-policy distillation. For GLM-5 it describes a final distillation pass with an earlier checkpoint as teacher, and for Nemotron 3 Ultra token-level guidance from specialist teachers on the student's rollouts, without using the term "on-policy"] Its practical advice is to prefer on-policy distillation, and to use an earlier or hint-conditioned checkpoint as a "self-teacher" to recover lost skills.
- **TRL implementations** (TRL v1.18.1, 17 Nov 2026):
  - `GKDTrainer` (GKD, Agarwal et al., arXiv 2306.13649): `lmbda` is the fraction of on-policy samples, `beta` runs from 0 (forward KL) to 1 (reverse KL), and `seq_kd` does SFT on teacher generations. The authors found high `lmbda` works better. The teacher is loaded in the same process.
  - `DistillationTrainer` (synchronous; the teacher is loaded locally or scored on a server) and `AsyncDistillationTrainer` (experimental). The async trainer needs the teacher server, the student's vLLM server and the trainer on **separate GPUs**. "Every teacher must share the student's tokenizer."
  - GOLD gained cross-tokenizer distillation ("X-Token") in TRL v1.17.0 (29 Oct, year inferred 2026). [corrected 2026-11-17: the GitHub API dates v1.17.0 to 2026-10-29, so the year is confirmed. The release notes call X-Token "off-policy cross-tokenizer distillation" (`lmbda=0.0` with a precomputed projection matrix). GOLD's older ULD mode (`use_uld_loss=True` plus `teacher_tokenizer_name_or_path`) is a separate route: it does cross-tokenizer distillation and inherits GKD's on-policy `lmbda` (default 0.5), per the GOLD docs. GOLD is in `trl.experimental`]
  - `SDFTTrainer` (Self-Distillation Fine-Tuning, arXiv 2601.19897). The teacher is the same model shown extra `privileged_context` ("a demonstration, hint, or privileged feedback"). It runs with PEFT: the teacher is the base weights with the adapter disabled, or an EMA held in a second adapter. Generation runs through transformers by default, with vLLM optional.

### Does on-policy distillation fit in 12 GB?

Only in reduced forms.

- **The teacher must expose log-probs on the student's tokens.** Closed APIs do not offer this, and their terms forbid it anyway (licence section). So the teacher must be an open-weight model with the same tokenizer, or one bridged by GOLD's cross-tokenizer mode. [corrected 2026-11-17: be precise about which GOLD mode. The new X-Token mode is off-policy only (`lmbda=0.0`, TRL v1.17.0 notes). GOLD's ULD mode (`use_uld_loss=True`) does align different tokenizers and keeps GKD's on-policy `lmbda` (default 0.5), so a different-tokenizer open teacher is possible on-policy through ULD. That route is approximate: it sorts and merges probabilities rather than matching tokens exactly. `AsyncDistillationTrainer` cannot do it: "Every teacher must share the student's tokenizer."]
- **Same-process teacher (GKD or synchronous DistillationTrainer).** A 4B student in 4-bit with LoRA plus a 9B teacher in 4-bit is roughly 3 + 6 GB of weights before activations and KV cache. That is tight at 12 GB with 4-8k sequences **[estimate]**.
- **The async trainer is out**: it needs three GPUs.
- **SDFT with a reference spec fits.** The teacher is the student itself, conditioned on a `privileged_context` column holding a validator-passing reference spec for that prompt. A refusal computed in advance for a typical bad spec can be added, for example "overflow in STAGE_LEFT: 9 bullets at MIN_FONT_SIZE". It needs no second model and no vLLM. [corrected 2026-11-17: "fits" is an **estimate**, not measured. With `teacher_model_kind="base"` under PEFT the teacher is the adapter-disabled base, so no second copy of the weights is needed, per the SDFT docs. The extra cost is a second forward pass over a longer teacher prompt (prompt plus reference spec). SDFT is in `trl.experimental`]
  - The column is fixed per prompt in the dataset, so a live refusal of the student's own sample cannot go there. A live-feedback teacher would need custom code **[inference]**.
  - This is the most laptop-friendly on-policy method found. Whether it beats plain RL on ChalkDust is **untested**.
- **Cloud option:** rent a 24-48 GB GPU for a few hours and run the synchronous DistillationTrainer with a same-family teacher, for example Qwen3.5-9B teaching Qwen3.5-4B.

### How much data for a narrow task?

No source gives a number for a narrow JSON-filling task like job 3. What exists:

- LIMA (arXiv 2305.11206): 1,000 curated examples were enough to align a 65B model for general chat, on the argument that most knowledge comes from pretraining.
- LoRA Land (arXiv 2405.00732): 310 4-bit LoRA fine-tunes (10 base models, 31 narrow tasks) beat their base models by 34 points and GPT-4 by 10 points on average.
- Yuan et al.: gains from supervised data grow log-linearly, so each doubling buys less.
- "LoRA Without Regret": LoRA matches full fine-tuning until the dataset outgrows its capacity, which small and medium datasets do not.

**Working estimate** (to be replaced by a learning curve):

| Job | Training examples to start | Notes |
|---|---|---|
| 3. Component params | 300-1,000 validator-passing examples per component, 5k-15k total | Keep 2-3 distinct passing samples per prompt where they differ |
| 2. Narration | 3k-5k beats | Only partly verifiable (capacity and duration rungs) |
| 1. Lesson outline | 1k-2k outlines | No verifier. Curate by hand or with a judge model; this is the job distillation helps least |

Measure validator pass@1 on a held-out set of topics at 1k, 3k and 10k examples before generating more.

## 2. LoRA, QLoRA and newer variants

**Short answer: no variant reliably beats a well-tuned plain LoRA. rsLoRA is worth switching on because it is free. DoRA is worth a try only if RL is the main stage. Avoid PiSSA-style initialisations for RL.**

### What "LoRA Without Regret" settled (Thinking Machines, John Schulman et al., 29 Sep 2025)

- On small-to-medium instruction and reasoning datasets, LoRA "performs the same as full fine-tuning". It falls behind only when the data exceeds LoRA's capacity.
- Apply LoRA to **all layers, especially the MLP layers**. Attention-only LoRA does worse even at matched parameter count.
- The best LoRA learning rate is about **10x** the full fine-tuning rate (about 15x for runs under ~100 steps). It barely changes with rank.
- LoRA copes with large batches worse than full fine-tuning, at any rank.
- For RL, LoRA "fully matches the learning performance of FullFT" even at **rank 1**.
- They used standard PEFT initialisation with alpha = 32 and "were unable to improve on these hyperparameters".

### Variants (PEFT docs and benchmarks)

| Variant | PEFT switch | What it claims | Independent evidence |
|---|---|---|---|
| rsLoRA | `use_rslora=True` | Scales by alpha/sqrt(r) so higher ranks train stably | HF "Beyond LoRA" (18 Jun 2026), Llama-3.2-3B on MetaMathQA to GSM8K: plain LoRA 48.1% at 22.5 GB, rsLoRA **53.2%** at 22.6 GB |
| DoRA | `use_dora=True` | Better "especially at low ranks"; more overhead, so merge before inference | PeRL (arXiv 2512.23165, Dec 2025): DoRA, AdaLoRA and MiSS consistently beat standard LoRA under RLVR |
| PiSSA, MiLoRA (SVD inits) | `init_lora_weights="pissa"` | Faster convergence | PeRL: under RLVR they **fail** ("spectral collapse"). SFT results for PiSSA vary between papers **[secondary]** |
| LoRA+ | `create_loraplus_optimizer` | Up to 2x faster, +1-2% | No independent check found |
| OLoRA, EVA, CorDA, LoftQ, QALoRA | various | Faster convergence, less forgetting, better quantised starts | QALoRA "only implemented in GPTQ for now" |
| Lily | — | — | Beyond LoRA: 54.9%, but at 25.6 GB |

- Beyond LoRA concludes that nothing beats LoRA on every axis. Its message is that LoRA is a sound choice but should not be picked automatically, not that a single winner exists.
- **Conflict:** Thinking Machines found rank-1 LoRA matches full fine-tuning in RL; PeRL found that extreme reductions (rank 1, VeRA) severely limit reasoning. Rank 16-32 costs little on a 4B model, so use that.
- **rsLoRA changes the effective scale.** With alpha = 32, standard LoRA scales updates by alpha/r = 2 at r = 16; rsLoRA scales by alpha/sqrt(r) = 8. The "10x learning rate" rule was found with standard scaling, so with rsLoRA on, re-sweep the learning rate (for example 2e-5 to 2e-4) rather than reusing it **[inference]**.

## 3. RL with verifiable rewards

### The algorithm family as of November 2026

| Method | What it changes | In TRL `GRPOConfig` (main docs) |
|---|---|---|
| GRPO (arXiv 2402.03300) | Group-relative advantages, no value model | `loss_type="grpo"`. The docstring says "Not recommended due to length bias" (quoted in HF's Aug 2026 post) [corrected 2026-11-17: the phrase is in the TRL `GRPOConfig` docstring itself, fetched this session] |
| DAPO (arXiv 2503.14476) | Token-level loss normalisation; Clip-Higher | `loss_type="dapo"`, **the default** |
| Dr. GRPO (arXiv 2503.20783) | Divides by a constant to remove length bias | `loss_type="dr_grpo"` |
| GSPO | Sequence-level importance ratio and clipping | `importance_sampling_level="sequence"` |
| SAPO (arXiv 2511.20347) | Soft gate instead of hard clipping | `loss_type="sapo"` |
| CISPO (ScaleRL, arXiv 2510.13786) [corrected 2026-11-17: TRL attributes CISPO to the MiniMax-M1 paper, arXiv 2506.13585; ScaleRL is not the source TRL cites] | Clips importance weights, not token updates | `loss_type="cispo"` |
| VESPO | Not described in the part of the docs read [corrected 2026-11-17: the `GRPOConfig` docstring describes it as "Variational Sequence-Level Soft Policy Optimization", a smooth asymmetric Gamma weighting of sequence-level importance weights instead of hard clipping, from arXiv 2602.10693 (paper not read)] | `loss_type="vespo"` (confirmed in the docstring) |
| RLOO | Leave-one-out baseline | Separate `RLOOTrainer` (named in the TRL release notes) |
| GMPO | Geometric-mean objective **[unverified; paper not read]** | A `GMPOTrainer` is named in TRL issue #7005 **[secondary: seen in search results only]** |
| REINFORCE++ (arXiv 2501.03262, v9 Nov 2025) | Critic-free like GRPO, but normalises advantages over the whole batch instead of per prompt group. The authors argue per-group normalisation is biased, and claim gains over GRPO and even PPO in agentic settings | Not a `GRPOConfig` option in the docs read; available in other frameworks **[unverified which]** |

- Hugging Face's August 2026 survey of frontier-lab reports says "GRPO is the usual starting point and each one changes something about it". It cites Magistral's rule to "filter out all groups with zero advantage".
- TRL v1.17.0 made zero-advantage filtering the default in AsyncGRPO. TRL v1.18.0 (13 Nov) made `AsyncGRPOTrainer` stable and added LoRA-only weight sync to vLLM.
- 2026 arXiv variants keep appearing: µ-GRPO 2605.17570, OAPL 2602.19362, calibration-aware 2604.12632, F-GRPO 2602.06717, and a survey 2606.16733. **[Seen in search results only; abstracts not read. No consensus successor.]**
- [corrected 2026-11-17: the `GRPOConfig` docstring also lists `loss_type="bnpo"` (normalises over active tokens in the local batch) and `loss_type="luspo"` (Length-Unbiased Sequence Policy Optimization, arXiv 2602.05261, needs `importance_sampling_level="sequence"`), which the table above omits]
- **Use:** `loss_type="dapo"` (the default), `beta=0.0` (no reference model, so less memory), zero-advantage filtering, `mask_truncated_completions=True`. Reward design matters far more than the loss for this task. [corrected 2026-11-17: zero-advantage filtering is an `AsyncGRPOTrainer` feature per the v1.17.0 notes, and the `GRPOConfig` docs show no switch for it in the synchronous `GRPOTrainer`. With the synchronous trainer, watch `frac_reward_zero_std` instead, or filter in the reward/data pipeline]

### Memory on a 12 GB laptop

- Unsloth's GRPO guide:
  - "Just 5GB VRAM is enough ... (for any model with 1.5B parameters or less)."
  - For QLoRA 4-bit, VRAM is roughly the parameter count in billions, plus more for longer context. "LoRA 16-bit will use at minimum 4x more VRAM."
  - Supports `loss_type` values `grpo`, `dr_grpo` and `gspo`.
- Unsloth's memory-efficient RL page: Qwen3-4B on a 16 GB T4 with Standby at `gpu_memory_utilization` 0.7 used about 13 GiB, usually 10-11 GB. Without Standby it used about 15.1 GiB, and 0.8-0.9 ran out of memory. Standby is built on vLLM sleep mode. [checked 2026-11-17: figures match the page. Precisely, Standby starts from vLLM's sleep mode but deletes only the KV cache and keeps the weights ("ignore deletion of the weights"). The page does not state LoRA/QLoRA, sequence length or generation count for these runs]
- TRL memory levers:
  - `beta=0.0`.
  - vLLM colocate mode with `vllm_gpu_memory_utilization` (default 0.3).
  - `vllm_enable_sleep_mode`.
  - The fused LM head, on by default since v1.15.0, which avoids building the full logits tensor. [corrected 2026-11-17: the v1.15.0 notes say "Scoring needs Triton on a GPU (Linux with CUDA, ROCm or XPU)". v1.16.0 then added "Run the fused LM head and distillation kernels on devices without Triton" (PR #7669), listed under the AWS Trainium (Neuron) section. So a non-Triton path exists, but it is documented for Neuron, and whether native Windows uses it or triton-windows is still untested. Measured anchor from the v1.15.0 notes: GRPO peak memory 10.30 GiB (down from 24.15) at 8,192 tokens. That run used `google/gemma-3-1b-pt` built from its config with random bf16 weights, synthetic token ids, AdamW, on a B300 capped at 79 GiB. The notes label the GRPO benchmark "scoring only", so generation memory is not included]
- **Estimate for machine A:**
  - 1.5-2B model: comfortable.
  - 4B model, QLoRA rank 16, 8 generations, 1,536-token completions, 4k prompts: **about 10-12 GB, at the edge** **[estimate from Unsloth's T4 figure]**. [corrected 2026-11-17: the Unsloth run's settings are not stated, so this extrapolation to 4k prompts and 8 generations is unsupported by any fetched source; treat it as a guess until the "Run when back" smoke test measures it]
  - 9B model: needs a cloud GPU.

### Windows blocker: fast generation

- RL time is mostly generation. Both Unsloth and TRL get fast generation from vLLM, and vLLM "does not support Windows natively" (vLLM docs). Unsloth says the same for GRPO with vLLM.
- Without vLLM, Unsloth's `fast_inference=False` and TRL's transformers path still run, but much slower **[slowdown not measured; Unsloth's fast_inference=False behaviour comes from a search snippet only]**.
- So RL on machine A should run in WSL2.

### Reward hacking when the reward is a validator

A validator checks *validity*, not *teaching value*. The cheapest ways to score high are the ones to block. The literature:

- Lilian Weng (28 Nov 2024): reward hacking is when an agent "exploits flaws or ambiguities in the reward function". Mitigations include capping the reward, combining several reward signals, and comparing behaviour against a trusted policy.
- "LLMs Gaming Verifiers" (arXiv 2604.15149, Apr 2026): RLVR models learned to pass a verifier that checked only surface correctness, by enumerating answers instead of learning the rule. A stricter verifier removed the shortcut.
- "Fuzzing RLVR Verifiers" (arXiv 2606.01066, May 2026): stress-test the verifier with adversarial completions *before* training. Parsing, schema validation and value comparison are each places for bugs. [corrected 2026-11-17: the abstract does not name parsing, schema validation or value comparison as bug sources; that sentence is this doc's inference. The abstract says "if the verifier is wrong, optimization can learn the bug". It proposes generating adversarial completions and comparing a verifier against a stricter reference, and it names three verifier kinds: math answer checkers, **JSON tool-call validators** (the closest analogue to ChalkDust) and code unit-test harnesses. Full title: "Before the Model Learns the Bug: Fuzzing RLVR Verifiers", submitted 31 May 2026]
- "Spurious Rewards" (arXiv 2506.10947): GRPO with *random* rewards raised Qwen2.5-Math-7B by 21.4 points on MATH-500, against 29.1 for true rewards. The effect did not hold for Llama3 or OLMo2. Treat RL gains on Qwen-family models with suspicion until a held-out evaluation confirms them.

ChalkDust-specific collapses to expect:

1. **Minimal specs.** One TitleCard, two short bullets: always valid, never overflows.
2. **Component bias.** Drifting to the components that never refuse (Callout, BulletReveal) instead of the one the beat needs (EquationDerivation, FreeBodyDiagram).
3. **RawScene abuse**, if it can pass checks the others fail.
4. **Narration padding or truncation** to satisfy the duration rung (8-25 s).
5. **Copying the prompt** into the params verbatim.
6. **Exploiting validator bugs**, for example a semantic check that skips a field.

### Combining a hard validator with a content floor

Proposed reward (design, untested):

- **Gate.** Failing rung 1 (schema) or rung 2 (semantic: capacity, duration, LaTeX, carry-in) gives reward **-1**. Failing the geometric probe after mechanical repair gives **-0.5**. Only passing samples go on to be scored.
- **Content floor**, scored only if the sample passes, summed and capped at +1:
  - **Coverage:** the fraction of the beat outline's required items (terms, equations, quantities) that appear in the params. This is the main anti-minimal-spec term.
  - **Component fit:** whether the chosen component is in the allowed set for the beat type, which the lesson plan already names. RawScene is penalised unless the outline asks for it.
  - **Size band:** penalise content below a minimum per component (for example EquationDerivation with fewer than 2 steps). The validator already caps the maximum.
  - **Faithfulness:** numbers and LaTeX in the params must match the outline or worked solution, checked by string or sympy comparison where possible.
- **Anchor to the SFT model.** Either keep a small KL (`beta` around 0.01-0.04) or, cheaper, log the component-choice distribution each eval step. Stop if it moves away from the SFT distribution on held-out prompts.
- **Hold out a final judge.** Never put the render check or a judge model in the training reward. Use them only to score held-out topics, so the model is measured on something it was not optimised against.

**Which validator rungs fit an RL inner loop:**

- The pydantic schema and the non-LaTeX semantic checks are cheap.
- LaTeX compilation (`check_latex`, MiKTeX) and the geometric probe build real objects. Measure their cost (Run when back) and cache by input hash, as the pipeline already does.
- Rendering is too slow for the training loop; keep it for evaluation.

## 4. Test-time compute

| Evidence | Finding | What it means for ChalkDust |
|---|---|---|
| "Large Language Monkeys" (Brown et al., arXiv 2407.21787) | Coverage (at least one correct sample) grows roughly log-linearly with samples over four orders of magnitude. With an automatic verifier this becomes real accuracy: SWE-bench Lite rose from 15.9% at 1 sample to 56% at 250. Without one, majority vote and reward models plateau | The validator is the automatic verifier. Best-of-N converts coverage straight into pass rate |
| Snell et al. (arXiv 2408.03314) | Compute-optimal allocation per prompt is over 4x more efficient than best-of-N. A small model with test-time compute can beat a 14x larger model where the small model has moderate success | Spend samples adaptively: stop at the first valid sample, sample more only on beats that fail |
| Liu et al. (arXiv 2502.06703) | With compute-optimal test-time scaling, a 1B model beat a 405B model on MATH-500, and a 3B model surpassed it on MATH-500/AIME24. The best strategy depends on the model, the verifier and difficulty | Small models plus a good verifier can punch far above their size on checkable tasks |
| Self-consistency (Wang et al., arXiv 2203.11171) | Majority vote over sampled reasoning paths: GSM8K +17.9% | Useful where an answer can be voted on, such as a numeric final answer in a worked solution |
| Self-repair (Olausson et al., arXiv 2306.09896, ICLR 2024) | Gains from self-repair are modest and inconsistent because models critique their own code badly. Stronger external feedback gave substantially larger gains | ChalkDust's repair loop should feed back the validator's **typed refusal** (kind, region, numbers), not ask the model to find its own error |

### Recommended inference policy

- **Job 3 (params):**
  - Constrained sample at temperature about 0.7.
  - On refusal, feed the refusal back once or twice (repair).
  - If it still fails, draw N fresh samples: up to 4 on A, 2-3 on B and C.
  - Pick the first sample that passes; break ties on the content-floor score.
- **Job 2 (narration):** sample 2-3 and keep those that pass the capacity and duration rungs and contain the beat's key terms.
- **Job 1 (outline):** no verifier, so best-of-N needs a judge. On B and C this costs too much. Rely on training quality, and offer a "regenerate outline" button.
- **Beam or tree search:** no evidence found that it beats verifier-guided sampling for short structured outputs. Whether llama.cpp's server supports beam search was not checked **[unverified]**. Not recommended for now.

## 5. Constrained decoding

### Engines

| Engine | State (Nov 2026) | Notes |
|---|---|---|
| llama.cpp GBNF / `json_schema` | Built in: `response_format` on `/chat/completions`, `json_schema` on completions. v1.0.0 (13 Nov 2026) fixed an `allOf`-over-plain-types bug, made several chat parsers (qwen3-coder, deepseek, ministral3, gemma4, Muse Spark) honour `response_format`, and accepts a top-level `response_format.schema` | "The JSON schema is only used to constrain the model output and is not injected into the prompt." Describe the schema in the prompt as well |
| llguidance (MIT) | "deemed v1.0.0" on 2025-06-23 | About 50 µs CPU per token mask with a 128k tokenizer. In llama.cpp via `-DLLAMA_LLGUIDANCE=ON`, in vLLM since v0.8.2, and in SGLang |
| XGrammar (Apache-2.0) | XGrammar-2 released 2026/05 [checked 2026-11-17: the README says "[2026/5] XGrammar-2 has been released!"; latest tagged release v0.2.9, 9 Nov 2026, per the GitHub API] | Used by vLLM, SGLang, TensorRT-LLM and MLC; `xgrammar[metal]` for Apple MPS. llama.cpp is not listed among its integrations |
| Outlines (Apache-2.0, dottxt) | No version shown on the repo page [corrected 2026-11-17: the GitHub API lists release 1.3.3, 6 Aug 2026, as the latest] | Backends include transformers, llama.cpp, vLLM and Ollama |

JSONSchemaBench (arXiv 2501.10868, Jan-Feb 2025) compares Guidance, Outlines, llama.cpp, XGrammar, OpenAI and Gemini on 10K real-world schemas plus the official JSON Schema Test Suite. The abstract gives no per-engine numbers (the full paper was not read). Its repo would let ChalkDust test its own component schemas against llama.cpp.

### Limits of llama.cpp's schema-to-grammar converter (`grammars/README.md`)

**"Unsupported features are skipped silently."** A schema can look enforced and not be. Known gaps, quoted from the README:

- "Can't mix `properties` w/ `anyOf` / `oneOf` in the same type"
- "prefixItems is broken (but items works)"
- "minimum, exclusiveMinimum, maximum, exclusiveMaximum: only supported for "type": "integer" for now, not number"
- "Nested `$ref`s are broken"
- patterns must start with `^` and end with `$`
- "Remote `$ref`s not supported in the C++ version"
- "`string` formats lack `uri`, `email`"
- "No patternProperties"
- Not planned: `uniqueItems`, `contains`/`minContains`, `not`, `if`/`then`/`else`, `dependentSchemas` [corrected 2026-11-17: the README calls these "unlikely to be implemented (hard and/or too slow to support w/ stateless grammars)", not "not planned", calls the list non-exhaustive, and also lists `$anchor`. The README also advises: "It is currently advised to use the command-line Python converter (see above) to see any warnings", which is how to catch the silent skips]

Other notes from the same README:

- `additionalProperties` defaults to `false`, which "produces faster grammars + reduces hallucinations".
- Repeating `x? x? x? ...` "may result in extremely slow sampling"; use `x{0,N}` instead.

**What this means for ChalkDust:**

- pydantic emits `$defs`/`$ref` for nested models and `anyOf` for optionals and unions. Both hit the gaps above.
- Generate one **flattened** schema per component: inline every `$ref`, and turn `Optional[X]` into a required field with a sentinel or a simple nullable type. Check that the converter's output grammar matches.
- Decode in two steps: pick the component first (an enum), then constrain to that component's params.
- Float bounds will not be enforced by the grammar, so the validator keeps checking them.
- The grammar removes syntax failures. The validator stays the authority.

### Does constraining hurt quality?

- **"Let Me Speak Freely?"** (Tam et al., arXiv 2408.02442, Aug 2024): "significant decline in LLMs reasoning abilities under format restrictions", worse with stricter formats.
- **dottxt's rebuttal "Say What You Mean"** (Will Kurt): with matched prompts, structured output was as good or better. Unstructured vs structured: GSM8K 0.77 vs 0.78, Last Letter 0.73 vs 0.77, Shuffled Objects 0.41 vs 0.44. It blames mismatched prompts, JSON prompts that never showed the schema, and conflating JSON mode with structured generation.
- **CRANE** (arXiv 2502.09061, ICML 2025): grammars that allow *only* the final answer reduce reasoning. Letting the grammar include a reasoning section recovers it, with up to +10 points on GSM-Symbolic and FOLIO.
- **"Thinking Before Constraining"** (arXiv 2601.07525, Jan 2026, EMNLP version May 2026): reason freely, then switch the constraint on after a trigger token. Up to 27% better than natural generation on their tasks.

**Consensus:** constraining the *answer* is fine; constraining away the *reasoning* hurts.

- Job 3 needs little reasoning, so constrain fully.
- For jobs 1-2, either generate free text and parse it, or put a short free-text `plan` field first in the schema.

### Best practices

- Show the exact schema, or a matching example, in the prompt. The grammar is not the prompt.
- Put any reasoning or plan field **before** the answer fields. dottxt used `{"reasoning": ..., "answer": ...}` with reasoning capped at 250 characters.
- Use enums for closed sets: component names, palette roles, regions.
- Use the model's chat template and end with an empty assistant turn.
- Keep `additionalProperties: false`, and bound arrays with `maxItems`.
- Train with the same schema and field order you decode with, so SFT data and inference match token for token.

## 6. Speculative decoding on consumer hardware

### What llama.cpp supports (`docs/speculative.md`, master, Nov 2026)

`--spec-type` takes:

| Type | What drafts the tokens |
|---|---|
| `draft-simple` | A separate small draft model; the docs call it "the most used approach" |
| `draft-eagle3` | An EAGLE-3 head that reads the target's hidden states; must be converted against a specific target |
| `draft-dflash`, `draft-dflash2`, `draft-dspark`, `draft-lilicorr` | Block-drafting heads |
| `draft-mtp` | The model's own multi-token-prediction heads |
| `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v` (experimental), `ngram-mod` | Draftless n-gram matching. `ngram-mod` uses about 16 MB; `--spec-default` turns it on |

llama.cpp v1.0.0 moved the NextN API from `llama-ext.h` into `llama.h`.

The docs list no separate "prompt lookup" type. Prompt-lookup decoding (drafting by copying n-grams from the prompt) is covered by the `ngram-*` modes.

### Measured speedups

- **EAGLE-3 in llama.cpp** (PR #18039, merged about 12 Jun 2026) on an RTX A6000 48 GB with draft size 8:

  | Target | Speedup |
  |---|---|
  | Llama-3.1-8B BF16 | 2.55-3.28x |
  | Llama-3.1-8B Q4_K_M | 1.62-2.26x (121.5 to 274.4 t/s at best) [corrected 2026-11-17: was "2.26x", the best of three prompts] |
  | Qwen3-8B BF16 | 1.62-2.17x [corrected 2026-11-17: was "2.17x"] |
  | Qwen3-14B | 1.26-1.46x [corrected 2026-11-17: was "1.46x"] |
  | Qwen3-32B Q4_K_M | 1.16-1.30x [corrected 2026-11-17: was "1.30x"] |
  | Qwen3-30B-A3B | 1.25-1.39x, on a DGX Spark 128 GB [corrected 2026-11-17: was "1.39x", and not measured on the A6000] |
  | gpt-oss-20B | 0.89-1.06x (no gain, sometimes slower), on a DGX Spark 128 GB [corrected 2026-11-17: not measured on the A6000] |
  | Llama-3.3-70B Q4_K_M | 1.85-2.41x [corrected 2026-11-17: row added; it was omitted] |

  Gains shrink for larger models and for MoE models.

  [corrected 2026-11-17: details from the PR page and the GitHub API.
  - Merge date confirmed: `merged_at` 2026-06-12.
  - Qwen3-30B-A3B and gpt-oss-20B were measured on an **NVIDIA DGX Spark 128 GB**, not the A6000.
  - Each row has three prompts (quicksort / Pythagorean / DC trip). The table originally gave the best of the three; it now gives the range. The 8B rows span 1.62-3.28x, not 2.2-3.3x.
  - Llama-3.3-70B Q4_K_M (1.85-2.41x) was omitted, and it does not fit "gains shrink for larger models". The shrinkage holds for the Qwen3 dense series, not across families.
  - The PR itself warns: "After refactoring, the performance data below may differ from current results."]
- **MTP:** one report gives 262 vs 225 tok/s (1.22x) with `draft-mtp`. A custom Qwen3.5-27B port on an M4 Max was *slower* (0.43x). **[Anecdotal; from search results, not fetched.]**
- **No numbers were found for an RTX 50-series laptop, an M1 or an iGPU.**

### What this means for ChalkDust

- **Job 3 is a near-ideal case for draftless n-gram speculation.** The output repeats field names and enum values, and copies text from the beat outline already in the prompt. `ngram-mod` or `ngram-simple` needs no extra model and no training. Try it first.
- **Qwen3.5-4B and 9B ship MTP heads**; their cards recommend MTP for speculative decoding. [corrected 2026-11-17: the cards say "MTP: trained with multi-steps" and give MTP-enabled launch commands for SGLang (`--speculative-algo NEXTN`) and vLLM (`qwen3_next_mtp`), but never use the phrase "speculative decoding". Gemma 4 E2B, E4B, 31B and 26B A4B also got MTP releases on 16 Apr 2026, per Google's Gemma release log] A LoRA fine-tune changes the hidden states the MTP head reads, so its acceptance rate may drop **[unverified]**. The same applies to EAGLE-3 heads, which would need re-training on the fine-tuned model.
- On B and C, generation is limited by memory bandwidth. Speculation helps in principle, but any draft model also uses bandwidth and RAM. Measure on each machine class before shipping.

## 7. Quantization

| Option | Use on | Evidence |
|---|---|---|
| k-quants (Q4_K_M, Q5_K_M, Q6_K) | All machines (CUDA, Vulkan, Metal, CPU) | llama.cpp quantize README, Llama-3.1-8B: Q4_K_M 4.89 bits/weight, 4.58 GiB, about 71.9 t/s; Q8_0 7.95 GiB, about 50.9 t/s; F16 14.96 GiB, about 29.2 t/s. Quality loss "can be minimized by using a suitable imatrix file" |
| i-quants (IQ4_XS, IQ3_*, IQ2_*) | Memory-starved machines | IQ4_XS 4.17 GiB vs Q4_K_M 4.58 GiB for 8B. No per-bit-width quality data for small models was found |
| SLM sensitivity | — | SLMQuant (arXiv 2511.13023) reports "fundamental disparities" between small and large models in quantization sensitivity, and finds that methods tuned for large models transfer poorly. The abstract names no bit widths. **No controlled study of JSON/schema adherence under quantization was found** |
| MXFP4 | Machine A; possibly others | llama.cpp PR #15091 ("llama : add gpt-oss", merged 5 Aug 2025) added an MXFP4 ggml type with "pretty much complete support for the CUDA, Vulkan, Metal and CPU backends". It is mainly used for models trained in MXFP4 (gpt-oss). Converting other models to it is not shown to beat Q4_K_M **[secondary]** |
| QAT checkpoints (Gemma 4) | B and C, if Gemma 4 is the base | Google (5 Jun 2026): Q4_0 QAT for Gemma 4, with "even higher overall quality compared to standard PTQ baselines" but no numbers. Unsloth: a naive conversion to llama.cpp Q4_0 matches the QAT weights only 24.77% byte-exactly. For 26B-A4B, Unsloth's dynamic quant scored 85.63% top-1 against 70.20% for naive Q4_0 |
| NVFP4 | Machine A only | llama.cpp PR #22196 (merged 28 Apr 2026): native Blackwell NVFP4 kernels. Perplexity on a GB10 with a 120B model: Q4_K 4.6237, NVFP4 W4Q8 4.6283, NVFP4 W4A4 4.6577. [checked 2026-11-17: numbers and merge date match. The model was Nemotron 3 Super 120B-A12B, the figures come from a reviewer comment, not the PR description, and the commenter noted that "quantizing activations to 4-bit undeniably hurts quality". The NVFP4 GGUF in that thread was made with the HF converter script, not `llama-quantize`] PyTorch 2.15 lists "Faster 4-bit (NVFP4) LLM decoding on NVIDIA Blackwell GPUs". Metal does not run NVFP4, and NVFP4 GGUFs cannot be made with `llama-quantize` **[both secondary sources]** |
| MLX (Apple) | Machine B | mlx-lm offers DWQ (distils the quantised model from the full one; slower than dynamic quantization but usually better; works best at 2-4 bits), AWQ, GPTQ and dynamic quantization. No quality numbers given |

**Recommendations:**

1. **Use ChalkDust's validator as the quantization metric.** Run validator pass@1 on held-out topics for BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M and IQ4_XS of the fine-tuned model. Ship the smallest quant within about 2 points of BF16. Perplexity says little about schema adherence.
2. **Build the imatrix from ChalkDust data** (the training prompts and accepted outputs), not generic text, so calibration matches the task **[reasonable but untested]**.
3. **Fine-tuning may erode a QAT checkpoint's advantage.** Google says the QAT weights can be fine-tuned directly with Transformers or Unsloth, but neither Google nor Unsloth says whether a plain BF16 LoRA fine-tune keeps the QAT quality benefit. Our guess is that it partly does not **[inference]**. PEFT's QALoRA (QAT-aware LoRA) supports only GPTQ. If Gemma 4 is the base, compare "fine-tune the QAT checkpoint, then quantize" against "fine-tune the normal checkpoint, then Q4_K_M" on the validator **[untested]**.
4. **Do not make NVFP4 a shipping format yet.** It helps only RTX 50-series owners and needs a separate conversion pipeline.

## 8. MoE expert pruning, merging and offloading

- **REAP** (Lasby et al., arXiv 2510.13999, ICLR 2026 [corrected 2026-11-17: the arXiv page names no venue, so "ICLR 2026" is **[unverified]**; v3 is dated 13 May 2026], v3 May 2026) scores experts using router gate values and expert activation norms. With 50% of experts pruned, Qwen3-Coder-480B and Kimi-K2 kept nearly all their code-generation performance. Pruning beat merging on generative tasks; the authors argue merging loses fine-grained control over expert routing. REAP uses activations, so it needs calibration data; ChalkDust's own prompts would be the natural choice **[inference]**.
- **AIMER** (arXiv 2603.18492, v4 30 Oct 2026): calibration-free pruning for 7B-47B MoEs. It scores all experts in 0.22-2.06 s and reports more balanced capability than other calibration-free methods.
- **Merge-prune hybrids and allocation methods** (REAM 2604.04356, EvoESAP 2603.06003) exist **[abstracts not read]**.
- **Offloading:**
  - `--n-cpu-moe N` and `-ot ".ffn_.*_exps.=CPU"` keep expert weights in RAM (October notes).
  - llama.cpp v1.0.0 added a host-memory MoE expert cache with a configurable CPU share (`moe_cache_size`, `moe_cache_cpu_share`).
  - The MoE cache is no longer applied to the draft model.

**For ChalkDust:** small dense models are the better fit for B and C. MoE work matters only if machine A, or the cloud tier, runs a mid-size MoE such as Gemma 4 26B-A4B (3.8B active) as a stronger local teacher or "quality mode". A domain-calibrated 50% REAP prune of such a model could fit 16 GB, but this is **speculative**: REAP's evidence is from much larger models.

## 9. Newer since October 2026, and other credible developments

| Development | Date | Relevance |
|---|---|---|
| llama.cpp v1.0.0: semantic versioning, NextN API in `llama.h`, FP8 KV cache (`-ctk/-ctv f8_e4m3`), host MoE expert cache, `response_format` fixes [corrected 2026-11-17: the release page does not mention semantic versioning; it links a discussion "Releases and versioning of `ggml-org` projects" that was not read, so "semantic versioning" is **[unverified]**. The other items match the release notes; GitHub API `published_at` 2026-11-13] | 13 Nov 2026 | Pin a v1.x release in the app. The FP8 KV cache saves memory on long contexts |
| TRL 1.17-1.18: AsyncGRPO stable; GOLD cross-tokenizer distillation; zero-advantage filtering default [corrected 2026-11-17: zero-advantage filtering is in `AsyncGRPOTrainer` only, and GOLD's new X-Token mode is off-policy. Also new: TRL 1.18 caps vLLM at `<0.34.0`] | 29 Oct to 17 Nov 2026 | The RL and distillation tooling is now mature enough for a small team |
| PyTorch 2.15 (CUDA 13.2 default) and triton-windows 3.9 | 28 Oct (year inferred 2026) [corrected 2026-11-17: GitHub API confirms 2026-10-28 for both] | Defines the native Windows stack (section 10) |
| AIMER v4 | 30 Oct 2026 | Calibration-free MoE pruning (section 8) |
| SDFT in TRL (arXiv 2601.19897) | 2026 | On-policy self-distillation with privileged context: the laptop-friendly distillation route |
| "Thinking Before Constraining" (arXiv 2601.07525) | Jan/May 2026 | Free reasoning followed by constrained output (section 5) |
| DiffusionGemma: 26B MoE, 3.8B active, Apache-2.0; up to 4x faster on dedicated GPUs (over 700 tok/s on an RTX 5090); quality below autoregressive Gemma 4; llama.cpp support still a draft PR (#24427) | 10 Jun 2026 | **Not now.** Too large for B and C, lower quality, no llama.cpp support |
| Latent reasoning, new small reasoning-distillation recipes | — | Nothing credible and specific to this use case was found in this session **[not covered]** |

## 10. Tooling on Windows 11 native vs WSL2 (machine A)

All rows are documented, not tested here.

| Tool | Native Windows 11 | WSL2 | Versions and facts |
|---|---|---|---|
| PyTorch | Yes | Yes | 2.15.0 (28 Oct, year inferred 2026): "CUDA 13.2 as the default wheel with CUDA 13.4 alongside it". Blackwell sm_120 needs a CUDA 12.8+ build. Check with `torch.cuda.get_arch_list()` [corrected 2026-11-17: the date is 2026-10-28 (GitHub API). The download.pytorch.org wheel indexes list torch 2.15.0 Windows wheels only under cu132 and cu134; the cu130 index stops at 2.14.1. Use cu132, since bitsandbytes' Windows CUDA wheels stop at 13.2. "sm_120 needs CUDA 12.8+" is **[unverified]** this session] |
| triton-windows | Yes (community, under the triton-lang org) | Use normal `triton` | v3.9.0-windows.post30 (28 Oct): "Triton 3.9 is only guaranteed to work with PyTorch 2.15"; pin `"triton-windows<3.10"`. v3.8.0.post28 fixed misaligned-address errors on sm120/121. Python 3.10 dropped |
| bitsandbytes | Yes, prebuilt | Yes | Windows x86-64 wheels for CUDA 12.8 (targets include sm100, sm120) and CUDA 13.0-13.2 (includes sm120). Needs Python ≥3.10 and PyTorch ≥2.4 |
| Unsloth | Yes: Conda, Python 3.12, `torch` from the cu130 index; or Docker image `unsloth/unsloth` (covers Blackwell) | Yes | The docs' sample output shows Torch 2.10.0+cu130 and Triton 3.6.0, labelled as example output. The docs say vLLM does not support Windows directly, so GRPO with vLLM needs WSL or Linux |
| vLLM | **No** ("vLLM does not support Windows natively"). Community forks exist (SystemPanic/vllm-windows) **[unofficial]** | Yes | Binaries built with CUDA 12.9 by default, plus 12.8 and 13.0. Compute capability 7.5+ [checked 2026-11-17: matches the docs, whose source links point at the v0.34.0 tag; the GitHub API shows v0.34.0 released 16 Nov 2026. TRL 1.18 supports only `vllm>=0.23.0,<0.34.0`, so the newest vLLM is too new for TRL; pin 0.33.x, which TRL's notes say requires torch 2.14.0] |
| TRL | Yes for SFT/DPO/KTO | Yes; needed for vLLM-based GRPO and distillation | v1.18.1 (17 Nov). The v1.17 release notes list transformers ≥5.22.0, peft ≥0.21.0 and accelerate ≥1.15.0 for their Mistral Large 4 example; v1.16 requires accelerate ≥1.10.1; v1.15 dropped Python 3.10. v1.14.0 ships a fused Triton log-prob and entropy kernel, on by default on CUDA, ROCm and XPU. Whether it falls back without Triton on native Windows is **[unverified]** [corrected 2026-11-17: v1.17.0 raised the general floor to `accelerate>=1.11.0` (PR #7797), so 1.10.1 is stale. On the Triton question: v1.15.0 says the fused LM head's "Scoring needs Triton on a GPU (Linux with CUDA, ROCm or XPU)", and v1.16.0 added "Run the fused LM head and distillation kernels on devices without Triton" (PR #7669). A fallback therefore exists, though the v1.16 notes place it under AWS Trainium support, so the question is only partly answered; which path native Windows with triton-windows takes is still to be tested] |
| llama.cpp | Yes (CUDA, Vulkan) | Yes | v1.0.0 (13 Nov 2026). Daily builds continue (b12474 on 17 Nov) |

### Version mismatches to resolve before installing

- Unsloth's Windows guide uses the **cu130** index. [checked 2026-11-17: that index carries torch only up to 2.14.1 for Windows]
- PyTorch 2.15's default wheel is **cu132**.
- TRL 1.18 needs **vLLM < 0.34**; vLLM 0.33 needs **torch 2.14.0** (added 2026-11-17).
- triton-windows 3.9 is tied to **torch 2.15**.
- Unsloth's example output shows **torch 2.10**.
- vLLM's default binaries use **CUDA 12.9**.

**Recommendation:**

- **Native Windows** (SFT and inference): one venv with Python 3.12, torch 2.15.0 from the **cu132** index, `triton-windows<3.10`, then bitsandbytes, then Unsloth or TRL. Record the exact versions that work. [corrected 2026-11-17: was "torch from the cu130 or cu132 index, the matching `triton-windows`". The cu130 index has no torch 2.15.0 Windows wheel (latest 2.14.1), and triton-windows 3.9 "is only guaranteed to work with PyTorch 2.15". The alternative is cu130 + torch 2.14.x + triton-windows 3.8, an inferred pairing, **[unverified]**. Avoid cu134 because bitsandbytes' Windows CUDA wheels stop at 13.2]
- **WSL2** (RL and distillation): a second environment where vLLM picks the torch version, and TRL is installed after it. [corrected 2026-11-17: pin `vllm<0.34` (for example 0.33.x, which needs torch 2.14.0 per TRL's notes), because TRL 1.18 does not support vLLM 0.34]
- Keep the two environments separate.

## Ranking by expected value for ChalkDust

Impact is the expected gain in validator pass rate and content quality. Feasibility is for training on machine A and inference on B and C.

| Rank | Technique | Jobs | Impact | Feasibility | Why |
|---|---|---|---|---|---|
| 1 | Validator best-of-N plus typed-refusal repair loop at inference | 3 (partly 2) | High | High: no training; cost is extra tokens, adaptive N | Verifier-guided sampling is where small models gain most (Brown et al., Snell et al.). External precise feedback beats self-repair (Olausson et al.) |
| 2 | Grammar-constrained decoding with flattened per-component schemas, plan field first where needed | 3 | High (removes syntax failures) | High: built into llama.cpp | Needs schema flattening because the converter skips features silently |
| 3 | Rejection-sampled SFT with LoRA from a permitted teacher | 1, 2, 3 | High | High: QLoRA 4B fits in 12 GB natively | The proven base step. Licence-clean teachers exist (DeepSeek API, Apache-2.0 open models) |
| 4 | Quantization chosen by validator pass rate (Q4_K_M plus ChalkDust imatrix; QAT if Gemma) | All | Medium-high: makes B and C possible | High | Cheap to measure; prevents silent quality loss |
| 5 | Prompt and schema hygiene (enums, field order, schema in prompt, same format in training and inference) | All | Medium | High | Free |
| 6 | N-gram speculative decoding (`ngram-mod`/`ngram-simple`) | 3 | Medium (speed) | High: no model | JSON repeats itself; needs measuring |
| 7 | GRPO/DAPO with validator gate plus content floor | 3 | Medium-high | Medium: WSL2, 12 GB tight for 4B, reward-hacking work | Use after SFT plateaus; hold out a judge |
| 8 | On-policy distillation (same-family teacher), or SDFT with a reference spec as privileged context | 1, 2, 3 | Medium-high | Medium-low on 12 GB (SDFT fits; GKD tight; async needs 3 GPUs) [corrected 2026-11-17: "SDFT fits" is an estimate, not measured] | The most compute-efficient post-SFT step per the evidence; consider cloud hours |
| 9 | rsLoRA (on); DoRA (try for RL) | All | Low-medium (+5 points in one SFT benchmark) | High | One flag |
| 10 | MTP or EAGLE-3 speculation | All | Medium (speed) | Low-medium: heads drift after fine-tuning | Revisit once the fine-tuned model is fixed |
| 11 | MoE pruning and offloading | Teacher or cloud tier | Low for B and C | Medium | Only if a mid-size MoE is used on A or in the cloud |
| 12 | Diffusion LMs | — | Unknown | Low | Too big, lower quality, no llama.cpp support |

## Proposed recipe

**Base:** a 4B-class dense model with an Apache-2.0 licence and a same-tokenizer bigger sibling. For example, Qwen3.5-4B with Qwen3.5-9B as the on-policy teacher. Both are `apache-2.0` on Hugging Face and ship MTP heads. Gemma 4 E4B with a larger Gemma 4 sibling also works. Final choice: models research.

### Stage 0: evaluation set (before any training)

- 150-300 held-out topics across JEE/CBSE physics and maths, first-year CS and chemistry.
- Metrics:
  - Validator pass@1 and pass@4 per job.
  - Repair-loop success rate.
  - Content-floor score.
  - A human or judge rating on 30 rendered videos.
- Run it on the stock model with the inference policy from section 4. This is the baseline everything must beat.

### Stage 1: data generation (rejection sampling)

- **Teacher:** one whose terms allow training a shipped model (licence section). The DeepSeek API allows it explicitly; a large Apache-2.0 open model also works.
- **Generation:** 4-8 samples per prompt. Keep those passing schema, semantic and geometric rungs. Render-check a 5% sample.
- **Dedupe**, and keep up to 3 distinct passing samples per prompt.
- **Volume (estimate):**
  - Job 3: 5k-15k examples
  - Job 2: 3k-5k
  - Job 1: 1k-2k, curated

### Stage 2: SFT (native Windows or WSL2, Unsloth or TRL)

| Setting | Value |
|---|---|
| Method | QLoRA 4-bit |
| LoRA | `target_modules="all-linear"`, r = 16 (job 3) to 32 (jobs 1-2), `lora_alpha=32`, `use_rslora=True` |
| Learning rate | Without rsLoRA: about 10x the model's full fine-tuning LR, roughly 1e-4 to 2e-4 for a 4B model **[the 10x rule is from Thinking Machines; the base LR is an assumption]**. With rsLoRA on, the effective scale is 4x larger at r = 16, so sweep 2e-5 to 2e-4 on a 500-example pilot first **[inference]** |
| Batch size | Modest effective batch, 16-32 sequences; LoRA dislikes large batches |
| Schedule | Cosine, 2-3 epochs, sequence length 4-8k |
| Adapters | One multi-task adapter with job tags, or one adapter per job. Measure both |
| Expected VRAM | About 5-8 GB for 4B at 4-8k context **[estimate; measure]** |
| Expected time | Hours, not days, for about 20k examples **[estimate]** |

### Stage 3a: RL on job 3 (WSL2, TRL `GRPOTrainer` or Unsloth, vLLM colocate with sleep or Standby)

| Setting | Value |
|---|---|
| Start from | The stage 2 adapter |
| Prompts | 1k-2k beat prompts not seen in SFT |
| GRPO settings | `loss_type="dapo"`, `beta=0.0` (or about 0.02 if component drift appears), `num_generations=8`, `max_completion_length` about 1,536, temperature about 1.0, `mask_truncated_completions=True` [corrected 2026-11-17: "zero-advantage filtering" removed from this list. It is documented only for `AsyncGRPOTrainer`, not `GRPOTrainer`; see section 3. With `GRPOTrainer`, monitor `frac_reward_zero_std`] |
| LoRA | r = 16-32 |
| Steps | 200-500 steps, evaluating every 50 on held-out topics; stop when the held-out pass rate stops rising or the component mix drifts **[estimate]** |
| Reward | Gate plus content floor (section 3). Fuzz the reward function before the first run |
| Expected VRAM | 4B about 10-12 GB (edge); 1.5-2B about 5-7 GB **[estimates from Unsloth figures]**. If 4B does not fit, run this stage on a rented 24 GB GPU [corrected 2026-11-17: the Unsloth T4 run does not state its LoRA, sequence-length or generation settings, so these numbers are guesses until measured (Run when back, item 5)] |

### Stage 3b (optional, instead of or after 3a): on-policy distillation

- **On the laptop:** SDFT with `privileged_context` set to a validator-passing reference spec, optionally with a refusal computed in advance for a typical bad spec. It is a fixed column, not live feedback.
- **In the cloud:** the synchronous DistillationTrainer with a same-family teacher.

### Stage 4: export

- Merge the adapter.
- Convert to GGUF.
- Build an imatrix from ChalkDust data.
- Quantize to Q6_K, Q5_K_M, Q4_K_M and IQ4_XS, and pick per machine class by validator pass@1 (section 7).
- MLX build for Macs: DWQ 4-bit, compared against the GGUF Q4_K_M under llama.cpp Metal.

### Stage 5: inference policy in the app

- Constrained decoding.
- Adaptive best-of-N with repair.
- `--spec-type ngram-mod` if it measures faster.
- FP8 KV cache on A if long contexts appear.

## Licences that matter for this plan

The four questions for each licence:

- **(a)** Is commercial use allowed?
- **(b)** Can a fine-tuned derivative ship in a paid, closed-source app?
- **(c)** May its outputs train another model that ships commercially?
- **(d)** What obligations come with it?

This is a reading of the fetched text, not legal advice.

| Licence / terms | (a) | (b) | (c) | (d) and operative clause |
|---|---|---|---|---|
| **Apache-2.0**: Qwen3.5-4B and 9B (HF card "License: apache-2.0"); Gemma 4 (HF card "Apache 2.0"; the "Gemma 4 license" link resolves to a plain Apache-2.0 page); DiffusionGemma; XGrammar; Outlines | Yes | Yes. §2, verbatim from apache.org: "each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form." Source need not be released | Yes. Apache-2.0 says nothing about outputs | §4(a): "You must give any other recipients of the Work or Derivative Works a copy of this License"; (b) mark modified files; (c) keep notices; (d) pass on any NOTICE file. §6: no right to use the licensor's trademarks, so do not brand the app with "Qwen" or "Gemma" beyond stating origin. **Gemma specifically:** the Gemma Terms of Use (last modified 1 Apr 2026) say "The terms below apply to Gemma models listed in the Appendix at bottom of this page. For Gemma 4 terms, see the Gemma 4 license." So the Gemma Terms and the Prohibited Use Policy that hangs off them govern earlier Gemma models, not Gemma 4 [checked 2026-11-17: the Gemma Terms scope sentence and last-modified date (1 Apr 2026) match. The Prohibited Use Policy page (last modified 21 Feb 2024) has no scope statement of its own; it applies to "Gemma or Model Derivatives", terms the Gemma Terms define. That it excludes Gemma 4 therefore follows from the Terms' scope note, **[inference]**] |
| **MIT**: llguidance | Yes | Yes | n/a | Keep the copyright and licence notice [checked 2026-11-17: the GitHub API reports llguidance as MIT and XGrammar and Outlines as Apache-2.0; Google's DiffusionGemma post says "Released under an Apache 2.0 license"; the Apache §2, §4 and §6 quotes in the row above match the apache.org plain text] |
| **Anthropic Commercial Terms** (page lists Jun 17, 2025) | Outputs are yours: "Anthropic hereby assigns to Customer its right, title and interest (if any) in and to Outputs." | n/a | **Restricted.** §D.4, verbatim: "Customer may not and must not attempt to (a) access the Services to build a competing product or service, including to train competing AI models or resell the Services except as expressly approved by Anthropic". Whether a small model inside a video app "competes" is a legal question; do not use Claude outputs as training data without counsel's sign-off | — |
| **Google Gemini API / AI Studio terms** (effective March 23, 2026) | Yes. "Google won't claim ownership over that content." | n/a | **Restricted.** "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)." | — |
| **OpenAI terms** | — | — | **[unverified]**: the services agreement and terms pages returned HTTP 403 this session. Check before using GPT outputs as training data [corrected 2026-11-17: the consumer Terms of Use (openai.com/policies/row-terms-of-use, "Effective: January 1, 2026", covering "ChatGPT, DALL·E, and OpenAI's other services for individuals") were fetched. Under what users may not do, they list, verbatim: "Use Output to develop models that compete with OpenAI." So treat OpenAI as **restricted**, like Anthropic and Google. The API Services Agreement still returned HTTP 403, so the business/API wording is **[unverified]**] | — |
| **DeepSeek Open Platform ToS** (released Apr 22, effective Apr 29, 2026) | Yes | n/a | **Allowed.** §4.2, verbatim: "Subject to your compliance to applicable law and our Terms, you have the following rights regarding the Inputs and Outputs of the Services: (1) You retain any rights, title, and interests—if any—in the Inputs you submit; (2) We assign any rights, title, and interests—if any—in the Outputs of the Services to you; (3) You may apply the Inputs and Outputs of the Services to a wide range of use cases, including personal use, academic research, derivative product development, training other models (such as model distillation), etc." | The preceding clause requires that Inputs and Outputs "will not violate our Terms, or any laws or regulations applicable to those Inputs and Outputs" and "will not infringe upon the legitimate rights and interests of others". §5.1: DeepSeek's models stay DeepSeek's property. No competing-products clause found on a page search. Prompts go to DeepSeek's servers, so send only synthetic topic prompts, never student data [checked 2026-11-17: §4.1-4.2 and both dates re-read verbatim from the page HTML. The Open Platform ToS says it "serves as a Specific Agreement to the 'DeepSeek Terms of Use'", so the parent Terms of Use (last update 27 Mar 2026) were also fetched. They repeat the distillation permission in §4.2(3), adding "as long as such usage is legal and adhere to these Terms". They contain no competing-model clause. The restrictions closest to this use: no "copying, transferring, leasing, lending, selling, or sub-licensing the entire or part of the Services" without authorisation; no reverse engineering or "attempting to discover the ... models, algorithms"; and no "capturing, copying any content of the Services, including but not limited to using any robots, spiders, or other automatic setups". Normal API use for synthetic data generation does not appear to fall under these, but that is a reading, not legal advice] |
| **DeepSeek V4 open weights** | MIT per secondary sources and the official model-card PDF, which could not be parsed **[unverified]** [corrected 2026-11-17: confirmed. The Hugging Face API gives `license: mit` for `deepseek-ai/DeepSeek-V4-Pro` and `deepseek-ai/DeepSeek-V4-Flash` (repos created 22 Apr 2026), and the Pro repo's LICENSE file is the standard MIT text ("Copyright (c) 2023 DeepSeek"). So (a) yes, (b) yes, (c) yes: MIT says nothing about outputs] | — | — | If self-hosted, confirm the LICENSE file on the Hugging Face repo first [corrected 2026-11-17: done. The MIT obligation is to include the copyright and permission notice "in all copies or substantial portions of the Software". These repos are very large MoE models (third-party reports say about 1.6T total parameters for Pro, **[unverified]**), so they are teacher candidates for the cloud, not the laptop] |

Also note:

- A third-party host serving an open model (an inference API) adds its own terms on top of the model licence **[not checked]**.
- Using Claude in this research session is not the same as training on Claude outputs. Nothing generated here should become training data.

## Run when back (needs the machine)

1. **Native Windows stack:**
   - `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"`. Confirm `sm_120` is listed.
   - `python -c "import triton; import bitsandbytes"`.
   - `python -c "import trl, trl.kernels"`, to answer the Triton fallback question.
2. **Validator timing per rung** on 100 typical specs: schema, semantic without LaTeX, LaTeX compile, geometric probe, render. Decides which rungs go in the RL reward.
3. **llama.cpp throughput** on A for a 4B Q4_K_M:
   - Plain.
   - `--spec-type ngram-mod`.
   - `--spec-type draft-mtp` (for Qwen3.5).
   - JSON-schema-constrained vs free.
   - Later repeat on an M1 16 GB and an iGPU laptop.
4. **Schema-to-grammar check:** run llama.cpp's converter on each component's flattened schema and diff the grammar against the schema, to catch silent skips.
5. **WSL2 RL smoke test:** TRL GRPO (or Unsloth) on a 1.5B model, 50 steps, `num_generations=8`. Record peak VRAM, then repeat with the 4B model.
6. **Quantization sweep** of the stage-2 model by validator pass@1 (section 7).

## Sources

Fetched this session:

1. Thinking Machines, On-Policy Distillation: https://thinkingmachines.ai/blog/on-policy-distillation/
2. Thinking Machines, LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
3. llama.cpp v1.0.0 release: https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0
4. llama.cpp releases: https://github.com/ggml-org/llama.cpp/releases
5. llama.cpp grammars README: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
6. llama.cpp speculative decoding docs: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/speculative.md
7. llama.cpp EAGLE-3 discussion: https://github.com/ggml-org/llama.cpp/discussions/15902
8. llama.cpp EAGLE-3 PR #18039: https://github.com/ggml-org/llama.cpp/pull/18039
9. llama.cpp NVFP4 PR #22196: https://github.com/ggml-org/llama.cpp/pull/22196
10. llama.cpp DiffusionGemma PR #24427: https://github.com/ggml-org/llama.cpp/pull/24427
11. llama.cpp quantize README: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
12. Unsloth GRPO and RL guide: https://unsloth.ai/docs/basics/reasoning-grpo-and-rl
13. Unsloth memory-efficient RL: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl
14. Unsloth Windows installation: https://unsloth.ai/docs/get-started/install/windows-installation
15. Unsloth Gemma 4 QAT: https://unsloth.ai/docs/models/gemma-4/qat
16. TRL GRPO Trainer docs: https://huggingface.co/docs/trl/main/en/grpo_trainer
17. TRL GKD Trainer docs: https://huggingface.co/docs/trl/main/en/gkd_trainer
18. TRL Async Distillation docs: https://huggingface.co/docs/trl/async_distillation_trainer
19. TRL SDFT docs: https://huggingface.co/docs/trl/main/en/sdft_trainer
20. TRL releases: https://github.com/huggingface/trl/releases
21. TRL kernels directory: https://github.com/huggingface/trl/tree/main/trl/kernels
22. HF blog, How frontier models train on outcomes in 2026: https://huggingface.co/blog/sergiopaniego/agentic-rl-2026
23. HF blog, Distillation in 2026 (so far): https://huggingface.co/blog/sergiopaniego/distillation-2026
24. HF blog, Beyond LoRA: https://huggingface.co/blog/peft-beyond-lora
25. PEFT LoRA developer guide: https://huggingface.co/docs/peft/main/en/developer_guides/lora
26. PyTorch releases: https://github.com/pytorch/pytorch/releases
27. PyTorch get-started page (fetched, but content was encoded and unusable): https://pytorch.org/get-started/locally/
28. triton-windows releases: https://github.com/triton-lang/triton-windows/releases
29. bitsandbytes installation: https://huggingface.co/docs/bitsandbytes/main/en/installation
30. vLLM GPU installation: https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
31. llguidance: https://github.com/guidance-ai/llguidance
32. XGrammar: https://github.com/mlc-ai/xgrammar
33. Outlines: https://github.com/dottxt-ai/outlines
34. dottxt, Say What You Mean: https://blog.dottxt.ai/say-what-you-mean.html
35. Let Me Speak Freely? (arXiv 2408.02442): https://arxiv.org/abs/2408.02442
36. JSONSchemaBench (arXiv 2501.10868): https://arxiv.org/abs/2501.10868
37. CRANE (arXiv 2502.09061): https://arxiv.org/abs/2502.09061
38. Thinking Before Constraining (arXiv 2601.07525): https://arxiv.org/abs/2601.07525
39. Large Language Monkeys (arXiv 2407.21787): https://arxiv.org/abs/2407.21787
40. Snell et al., compute-optimal test-time scaling (arXiv 2408.03314): https://arxiv.org/abs/2408.03314
41. Can 1B LLM Surpass 405B LLM? (arXiv 2502.06703): https://arxiv.org/abs/2502.06703
42. Self-consistency (arXiv 2203.11171): https://arxiv.org/abs/2203.11171
43. Is Self-Repair a Silver Bullet? (arXiv 2306.09896): https://arxiv.org/abs/2306.09896
44. PeRL, PEFT methods for RLVR (arXiv 2512.23165): https://arxiv.org/abs/2512.23165
45. LoRA Land (arXiv 2405.00732): https://arxiv.org/abs/2405.00732
46. Rejection sampling fine-tuning, Yuan et al. (arXiv 2308.01825): https://arxiv.org/abs/2308.01825
47. LIMA (arXiv 2305.11206): https://arxiv.org/abs/2305.11206
48. Spurious Rewards (arXiv 2506.10947): https://arxiv.org/abs/2506.10947
49. Lilian Weng, Reward Hacking: https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
50. LLMs Gaming Verifiers (arXiv 2604.15149): https://arxiv.org/abs/2604.15149
51. Fuzzing RLVR Verifiers (arXiv 2606.01066): https://arxiv.org/abs/2606.01066
52. Google, Gemma 4 QAT: https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/
53. Google, DiffusionGemma: https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/
54. mlx-lm learned quantization: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md
55. SLMQuant (arXiv 2511.13023): https://arxiv.org/abs/2511.13023
56. REAP (arXiv 2510.13999): https://arxiv.org/abs/2510.13999
57. AIMER (arXiv 2603.18492): https://arxiv.org/abs/2603.18492
58. Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
59. Qwen3.5-4B model card: https://huggingface.co/Qwen/Qwen3.5-4B
60. Gemma 4 E4B QAT model card: https://huggingface.co/google/gemma-4-e4b-it-qat-q4_0-unquantized
61. Gemma licence page: https://gemma.ai/resources/apache-2/
62. Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0
63. Anthropic Commercial Terms: https://www.anthropic.com/legal/commercial-terms
64. Google AI Studio / Gemini API terms: https://aistudio.google.com/docs/terms
65. DeepSeek Open Platform Terms of Service: https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html
66. Apache License 2.0, plain text (for verbatim quotes): https://www.apache.org/licenses/LICENSE-2.0.txt
67. Gemma Terms of Use (scope note excludes Gemma 4): https://gemma.ai/resources/terms/
68. Gemma Prohibited Use Policy: https://gemma.ai/resources/prohibited-use/
69. REINFORCE++ (arXiv 2501.03262): https://arxiv.org/abs/2501.03262
70. llama.cpp gpt-oss / MXFP4 PR #15091: https://github.com/ggml-org/llama.cpp/pull/15091

Attempted but not readable (claims that depend on them are marked [unverified]):

- OpenAI services agreement (HTTP 403): https://openai.com/policies/services-agreement/
- OpenAI terms of use (HTTP 403): https://openai.com/policies/row-terms-of-use/ [corrected 2026-11-17: fetched successfully in the verification pass; see source 71]
- DeepSeek V4 model card PDF (could not be parsed): https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf

Seen only in search results, not fetched (claims marked [secondary] or [anecdotal]):

- NVFP4 secondary guides, including the Metal limitation and the `llama-quantize` limitation: https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/
- MTP speedup report: https://github.com/gittensor-ai-lab/sparkinfer/issues/508
- Qwen3.5-27B MTP port: https://github.com/NodeNestor/qwen3.5-27b-mtp-llamacpp
- Unsloth advanced RL docs (`fast_inference=False`): https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation
- REAM: https://arxiv.org/abs/2604.04356
- EvoESAP: https://arxiv.org/pdf/2603.06003v2
- TRL issue #7005 (GMPOTrainer): https://github.com/huggingface/trl/issues/7005
- 2026 GRPO variants: µ-GRPO https://arxiv.org/pdf/2605.17570, OAPL https://arxiv.org/pdf/2602.19362, calibration-aware PO https://arxiv.org/html/2604.12632, F-GRPO https://arxiv.org/pdf/2602.06717, survey https://arxiv.org/abs/2606.16733
- MXFP4 secondary guide: https://insiderllm.com/guides/fp4-inference-llamacpp-nvfp4-mxfp4/
- PiSSA SFT results in other papers: https://arxiv.org/pdf/2505.16531

Added by the 2026-11-17 verification pass (fetched in that session):

71. OpenAI Terms of Use (consumer, effective 1 Jan 2026): https://openai.com/policies/row-terms-of-use/
72. TRL releases via the GitHub API (dates and notes for v1.13.0 to v1.18.1): https://api.github.com/repos/huggingface/trl/releases
73. GitHub API, llama.cpp PRs #18039, #22196, #15091, #24427 (merge dates and state): https://api.github.com/repos/ggml-org/llama.cpp/pulls/18039 (same path for the other numbers)
74. GitHub API, PyTorch latest release: https://api.github.com/repos/pytorch/pytorch/releases/latest
75. GitHub API, triton-windows releases: https://api.github.com/repos/triton-lang/triton-windows/releases
76. GitHub API, llama.cpp releases and v1.0.0: https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/v1.0.0
77. PyTorch wheel indexes (which torch versions exist per CUDA build): https://download.pytorch.org/whl/cu130/torch/, https://download.pytorch.org/whl/cu132/torch/, https://download.pytorch.org/whl/cu134/torch/
78. Hugging Face API and LICENSE, DeepSeek-V4: https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro, https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash, https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/LICENSE
79. GitHub API licence fields: https://api.github.com/repos/guidance-ai/llguidance, https://api.github.com/repos/mlc-ai/xgrammar, https://api.github.com/repos/dottxt-ai/outlines (plus their /releases)
80. Gemma release log (redirect target of ai.google.dev/gemma/docs/releases): https://gemma.ai/models/releases/
81. Unsloth Qwen3.6 page: https://unsloth.ai/docs/models/qwen3.6
82. "Evaluating Parameter Efficient Methods for RLVR" (PeRL) and other arXiv abstracts re-read: https://arxiv.org/abs/2512.23165, https://arxiv.org/abs/2510.13999, https://arxiv.org/abs/2603.18492, https://arxiv.org/abs/2502.09061, https://arxiv.org/abs/2601.07525, https://arxiv.org/abs/2604.15149, https://arxiv.org/abs/2606.01066, https://arxiv.org/abs/2408.03314
83. MiniMax-M1 paper, cited by TRL for CISPO (not read): https://huggingface.co/papers/2506.13585
84. VESPO and LUSPO papers, cited by TRL (not read): https://huggingface.co/papers/2602.10693, https://huggingface.co/papers/2602.05261
85. TRL GOLD Trainer docs (ULD on-policy cross-tokenizer mode; privileged context off-policy only): https://huggingface.co/docs/trl/main/en/gold_trainer
86. DeepSeek Terms of Use (parent agreement of the Open Platform ToS, last update 27 Mar 2026): https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html
87. vLLM releases (GitHub API) and PyPI (0.33.0 on 2 Nov, 0.34.0 on 16 Nov 2026): https://api.github.com/repos/vllm-project/vllm/releases, https://pypi.org/pypi/vllm/json
88. Gemma Prohibited Use Policy, re-fetched by curl (last modified 21 Feb 2024): https://gemma.ai/resources/prohibited-use/

Seen only in search results during the verification pass (not fetched):

- Qwen3.6-35B-A3B model card (licence `apache-2.0` in the snippet): https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md
- No Gemma 5 announcement found; Wikipedia summary of Gemma releases: https://en.wikipedia.org/wiki/Gemma_(language_model)
- DeepSeek-V4 size reports (about 1.6T total for Pro): https://lilting.ch/en/articles/deepseek-v4-preview-1m-context-csa-hca-moe

## Verification

Adversarial check run on 17 November 2026 by a second research session. Every decision-critical claim (1-15 in the author's list) and every licence statement was re-fetched from its cited source. The release pages were also checked for anything newer. Corrections are inline, marked "[corrected 2026-11-17: ...]"; confirmations that add detail are marked "[checked 2026-11-17: ...]".

**Decision-critical claims:**

| # | Claim | Result |
|---|---|---|
| 1 | vLLM not native on Windows; WSL2 for GRPO | Confirmed (vLLM and Unsloth docs). Added: TRL 1.18 supports only `vllm<0.34`, but vLLM 0.34.0 shipped on 16 Nov 2026 (GitHub API, PyPI); pin 0.33.x (PyPI, 2 Nov 2026) |
| 2 | llama.cpp schema converter skips silently; gaps list | Confirmed quote by quote. Fixed: "not planned" is really "unlikely to be implemented"; `$anchor` was missing; the README recommends the Python converter to see warnings |
| 3 | TRL `loss_type` defaults to `dapo`; `beta=0.0` skips the reference model | Confirmed. Fixed: CISPO's cited source (MiniMax-M1, not ScaleRL); VESPO now described; `bnpo` and `luspo` were missing; zero-advantage filtering is AsyncGRPO-only |
| 4 | Anthropic Commercial Terms D.4 | Confirmed verbatim from the page HTML; effective date Jun 17, 2025; no newer version found |
| 5 | Gemini API terms, competing models | Confirmed verbatim; effective March 23, 2026 |
| 6 | DeepSeek ToS §4.2 allows distillation | Confirmed verbatim from the page HTML; released 22 Apr, effective 29 Apr 2026; no competing-use clause found. The parent DeepSeek Terms of Use (27 Mar 2026) repeat the permission and add no competing-model clause; their anti-copying and anti-scraping restrictions are noted in the licence table |
| 7 | Qwen3.5-4B/9B Apache-2.0 with MTP | Confirmed. Fixed: the cards do not use the phrase "speculative decoding" |
| 8 | Gemma 4 Apache-2.0; Gemma Terms exclude Gemma 4 | Confirmed. The "Gemma 4 license" page is plain Apache-2.0 text. The Prohibited Use Policy has no scope statement of its own, so its exclusion of Gemma 4 is inferred from the Terms |
| 9 | AsyncDistillationTrainer needs separate GPUs and a shared tokenizer | Confirmed verbatim (TRL v1.18.1 docs) |
| 10 | On-policy distillation 60% to 70% AIME'24 in about 150 steps; 1,800 vs 17,920 GPU hours | Confirmed |
| 11 | LoRA Without Regret findings | Confirmed. Note: the "same as full fine-tuning" claim is about SFT datasets; the rank-1 claim is about RL |
| 12 | EAGLE-3 speedups on RTX A6000 | **Partly wrong.** The MoE rows were measured on a DGX Spark, the single figures are best-of-three-prompts, a 70B row was omitted, and the PR warns its numbers predate a refactor. Merge date confirmed as 12 Jun 2026 |
| 13 | Unsloth Qwen3-4B T4 memory figures | Numbers confirmed. Fixed: the run's LoRA/sequence/generation settings are unstated, so the 4k-prompt extrapolation is unsupported |
| 14 | Large Language Monkeys 15.9% to 56%; self-repair modest | Confirmed (DeepSeek-Coder-V2-Instruct; feedback from stronger models or humans helps more) |
| 15 | PyTorch 2.15 cu132 default; triton-windows 3.9 tied to 2.15; bitsandbytes sm120 | Quotes confirmed. **Fixed:** the cu130 index has no torch 2.15.0 Windows wheel, so the "cu130 or cu132" advice was inconsistent with triton-windows 3.9. Recommendation changed to cu132 (not cu134, which bitsandbytes' Windows wheels do not cover) |

**Licences:** Apache-2.0 §2, §4 and §6 checked against the apache.org text. Qwen3.5, Gemma 4, XGrammar, Outlines and DiffusionGemma confirmed Apache-2.0. llguidance confirmed MIT. DeepSeek-V4 weights confirmed MIT (LICENSE file), which changes that row from [unverified] to confirmed. OpenAI consumer Terms of Use fetched and found to forbid "Use Output to develop models that compete with OpenAI"; the API Services Agreement is still blocked (HTTP 403).

**Other claims spot-checked (all matched their source unless corrected inline):** Yuan et al. RFT (35.9% to 49.3%); LoRA Land (310 models, +34 and +10 points); Spurious Rewards (21.4 vs 29.1 points, not Llama3/OLMo2); Liu et al. 1B vs 405B; Snell et al. (more than 4x, 14x larger model); Beyond LoRA (48.1% at 22.5 GB, 53.2% at 22.6 GB, Lily 54.9% at 25.6 GB); PeRL (DoRA/AdaLoRA/MiSS beat LoRA; spectral collapse; rank 1 and VeRA bottleneck); HF "Distillation in 2026" (fixed: GLM-5 and Nemotron not called on-policy); HF "How frontier models train on outcomes" quotes; SDFT docs; GOLD docs (fixed: X-Token and privileged context are off-policy only, but GOLD's ULD mode does cross-tokenizer distillation on-policy, so a different-tokenizer teacher is possible through ULD); llama.cpp `speculative.md` spec types and `ngram-mod` about 16 MB; quantize README numbers; NVFP4 PR #22196 (added the model name); MXFP4 PR #15091 merge date; DiffusionGemma (26B MoE, 3.8B active, Apache-2.0, above 700 tok/s on an RTX 5090; PR #24427 still an open draft); Gemma 4 QAT blog (5 Jun 2026; fine-tune with Transformers or Unsloth); Unsloth QAT (24.77%, 85.63% vs 70.20%); Unsloth GRPO guide (5 GB, loss types); llguidance (v1.0.0 on 2025-06-23, about 50 µs, vLLM 0.8.2); XGrammar-2 (2026/5); dottxt (0.77/0.78, 0.73/0.77, 0.41/0.44, 250 characters); CRANE (ICML 2025, up to 10 points); Thinking Before Constraining (up to 27%, v2 28 May 2026); REAP (fixed: venue not on arXiv); AIMER (v4 30 Oct 2026, 0.22-2.06 s); LLMs Gaming Verifiers; Fuzzing RLVR Verifiers (fixed: bug-source list was not in the abstract); TRL v1.13-v1.18.1 release notes (fixed: the accelerate floor; Triton fallback partly answered, since the v1.16 non-Triton path is documented under Trainium support); llama.cpp v1.0.0 notes (fixed: "semantic versioning" not on the page); the October notes (`chalkdust-research-2026-10.md`, partial) do contain the `--n-cpu-moe N` and `-ot ".ffn_.*_exps.=CPU"` flags this doc attributes to them.

**Newer-than-the-doc checks:** Qwen3.6 exists (27B and 35B-A3B; no small sizes found). Gemma 4 gained MTP heads and a 12B Unified model; no Gemma 5 was found. TRL 1.18.1 (17 Nov 2026), PyTorch 2.15.0, triton-windows 3.9.0.post30 and llama.cpp v1.0.0 are the latest releases as of this pass (llama.cpp daily builds had reached b12475 on 17 Nov).

**Still unverified after this pass:** the OpenAI API Services Agreement wording; "semantic versioning" for llama.cpp v1.0.0; REAP's ICLR 2026 venue; "sm_120 needs CUDA 12.8+"; which TRL scoring path native Windows takes; the triton-windows 3.8 / torch 2.14 pairing; the Gemma Prohibited Use Policy's exclusion of Gemma 4 (inferred, not stated); DeepSeek-V4 parameter counts; every VRAM estimate in the doc, including "SDFT fits in 12 GB" (none is measured); the GMPO, REINFORCE++ framework, MTP-anecdote and Unsloth `fast_inference=False` items the author already tagged as secondary or unverified (not re-checked).
