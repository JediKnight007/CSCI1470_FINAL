# Check-in 2 Reflection
**Team: Ube Macchiatos**

---

## Introduction

MambaVision is a recently proposed hybrid architecture that combines state space models (SSMs)
with transformer-like components to improve the tradeoff between accuracy and efficiency. It
introduces a mixer block that integrates an SSM branch for efficient sequence modeling and a
symmetric non-SSM bypass branch for complementary representational capacity. In this project,
we perform a critical replication of MambaVision-T, focusing on verifying its claimed advantages
over standard vision transformers. We evaluate both MambaVision-T and ViT baselines on STL-10
and compare their performance in terms of accuracy and computational efficiency. Beyond standard
evaluation, we design hypothesis-driven ablation studies to understand the contribution of the
mixer block's individual components. Our goal is to examine not only whether MambaVision performs
well, but also when its advantages hold and where they may break down.

---

## Challenges

**Fixing NaN Errors:**
A major challenge was dealing with NaN errors caused by gradient explosion and numerical
instability in the SSM components, where small errors accumulate to cause floating-point overflow.
We carefully tuned hyperparameters including learning rate, warmup schedule, and drop path rate
to stabilize training.

**Untangling the Mixer Block:**
For ablation studies, we needed to isolate specific model components. The original mixer block
splits the input into an SSM branch and a non-SSM bypass branch, each operating on half the
channel dimension before merging. Safely removing individual branches without breaking the
model's structural integrity or test fairness required careful surgery on the forward pass and
was our most difficult coding task.

---

## Results

Table 1 summarizes all runs. All MambaVision variants are trained from scratch on STL-10
(25,000 augmented images) on Brown University's OSCAR cluster (NVIDIA RTX 3090).
Deltas for MambaVision ablations are relative to the full MambaVision-T baseline.
ViT models are trained under identical data conditions but are a separate architectural
comparison — their deltas are shown for reference only.

**Table 1: Full Results Summary**

| Model                              | Params  | Best Top-1 | Epoch | Delta vs Baseline      |
|------------------------------------|---------|------------|-------|------------------------|
| MambaVision-T (full)               | 31.8M   | **89.225%**| 290   | —                      |
| MambaVision-T (no bypass)          | ~31.8M  | 87.625%    | 280   | -1.600%                |
| MambaVision-T (first-half attn)    | 31.8M   | 87.950%    | 270   | -1.275%                |
| MambaVision-T (no attn)            | 31.8M   | 85.375%    | 259   | -3.850%                |
|                                    |         |            |       |                        |
| ViT-Tiny *(separate baseline)*     | 5.7M    | 71.310%    | 350   | -17.915% *(ref only)*  |
| ViT-Small *(separate baseline)*    | 22M     | 68.390%    | 298   | -20.835% *(ref only)*  |

---

## Insights

Our results validate the original paper's claims and yield concrete insights about what drives
MambaVision's performance. Our replicated MambaVision-T baseline achieved a best Top-1 accuracy
of **89.225%** on STL-10, substantially outperforming both ViT baselines.

> **Note on ViT-Small vs ViT-Tiny:** ViT-Small (22M params) scoring lower than ViT-Tiny (5.7M)
> is consistent with overfitting on a small dataset. With only 25,000 training images, the larger
> model does not generalize as well despite its higher capacity.

Our three ablations yielded the following findings:

**1. Attention Placement Matters (-1.275%)**
Moving self-attention to the first half of the network stages caused a 1.275% accuracy drop
(to 87.95%), confirming that self-attention is most effective in the final layers where global
context aggregation is most needed. Placing it early, before rich features have been built up,
reduces its effectiveness.

**2. The Bypass Branch is Equally Critical (-1.600%)**
Removing the symmetric non-SSM bypass branch caused the largest single-component drop of 1.6%
(to 87.625%). Without this branch, the mixer block loses a complementary representational
pathway — the SSM output alone has no parallel signal to merge with, reducing the block's
overall capacity. Crucially, this penalty is *larger* than the attention placement penalty,
making it our most significant finding.

**3. Removing Attention Entirely is Most Damaging (-3.850%)**
Replacing all attention blocks with SSM-only mixers caused the largest overall drop of 3.85%
(to 85.375%), confirming that self-attention in the final stages is not merely helpful but
essential to MambaVision's performance.

---

## Plan

**Are we on track?**
Yes. All four model variants: full baseline, no bypass, first-half attention, and no attention, 
have completed training and results are confirmed. Our next steps are to complete stress
testing under challenging conditions (extreme aspect ratios, low-data regimes, and distribution
shifts) and to write up the final analysis.

**Narrative Adjustment:**
We are revising our conclusion based on the final results. Initially we expected attention
placement to be the dominant factor. However, the no-bypass ablation produced a steeper accuracy
penalty (-1.6%) than shifting attention blocks (-1.275%). Our revised conclusion emphasizes that
MambaVision's performance depends equally on its internal block design, specifically the bypass
branch, as on its macro-level attention placement strategy. The two design choices are
complementary, and removing either one meaningfully degrades accuracy.

