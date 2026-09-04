
- **Adaptive retraining breaks it.** If the scraper *knows* cloaked images are in their
  training set (or just retrains periodically), the recognition model adapts to the
  perturbation distribution and accuracy recovers. This is literally how Fawkes was broken
  in follow-up work.
- **Poor transferability.** A perturbation optimized against one embedding model (say,
  a specific FaceNet checkpoint) often doesn't transfer to a different architecture
  (ArcFace, CosFace, a CLIP-based retrieval model, a commercial API). Real-world scrapers
  use models you don't have white-box access to.
- **Wrong threat model for 2025+.** The bigger current risk isn't "does a classifier
  misidentify me" — it's "can someone LoRA/DreamBooth fine-tune a diffusion model on my
  photos to generate fake images of me." That's a different attack surface than embedding-space
  cloaking and needs different defenses (PhotoGuard, Anti-DreamBooth, Mist, MetaCloak-style
  work).

Any "more powerful" version should explicitly target one or more of these failure modes,
not just turn up epsilon on FGSM.

---

## A. Directions for a stronger protection tool

### 1. Attack the right threat model
Pick (or support multiple) explicit threats instead of one generic "fool a classifier":
- **Face verification/identification** (traditional cloaking — Fawkes/LowKey territory).
- **Generative fine-tuning protection** (stop DreamBooth/LoRA/textual-inversion from
  learning your likeness) — this is the more relevant threat for deepfake-style misuse today.
- **Embedding-based retrieval/scraping** (CLIP-style image search, "reverse image search
  for people") — different loss surface than either of the above.

Being explicit about which threat you're defending against — and saying which ones you're
*not* — is itself a strength, not a weakness, of the writeup.

### 2. Improve transferability
- Optimize the perturbation against an **ensemble** of diverse embedding models
  (FaceNet, ArcFace, dlib, a CLIP vision encoder) simultaneously rather than one model.
  Perturbations that fool several architectures at once transfer much better to unseen
  black-box models.
- Add **Expectation over Transformation (EOT)**: during optimization, apply random
  resize/JPEG-compress/crop/color-jitter and optimize the perturbation to survive the
  *expected* transformed image, not just the raw one. This is the standard technique for
  making adversarial examples survive real-world pipelines (originally from physical-world
  adversarial patch research) and directly targets the "social media re-compression kills it"
  failure mode.

### 3. Make it robust to purification specifically
- Explicitly evaluate against (and optimize against) a diffusion purification step in the
  loop — i.e., treat "attacker runs DiffPure before recognition" as part of the threat model,
  not an afterthought.
- Look at frequency-domain placement of the perturbation (e.g., mid-frequency bands survive
  denoising/purification better than high-frequency pixel noise, which purifiers are
  specifically tuned to remove).

### 4. Go semantic instead of purely pixel-level
- Instead of imperceptible epsilon-ball noise, consider **identity-shifting perturbations**
  that push the face embedding toward a specific decoy (a synthetic average face, or another
  consenting identity) rather than just off-manifold. Targeted attacks tend to be more robust
  than untargeted ones because there's a consistent objective for the optimizer to converge on.
- Look at GAN-based face de-identification (CIAGAN, DeepPrivacy-style work) as a stronger
  alternative/complement to gradient perturbation — these modify the image more substantially
  in ways that preserve *human* perceptual identity while destroying *machine* embedding
  similarity, which is a fundamentally different (and often more durable) trade-off than tiny
  pixel noise.

### 5. Data-poisoning angle (Nightshade-style, applied to identity)
- Rather than (or in addition to) protecting a single photo at inference time, optimize
  perturbations so that if a model *does* train on them, it learns a corrupted association —
  e.g., poisons a fine-tune so generated images of "you" are visibly wrong. This shifts the
  cost onto anyone who scrapes and trains, not just onto inference-time recognition.

### 6. Certified / provable robustness angle
- Randomized smoothing can give a certified radius within which the perturbation is
  guaranteed to hold, instead of only empirical "we tested it against these three attacks."
  This is a meaningfully deeper technical angle than typical PGD/FGSM cloaking tools bother
  with, and it's honest about what "protected" actually means (a probabilistic guarantee vs.
  "we didn't happen to break it").

### 7. Non-adversarial-ML defense layer (complementary, not a replacement)
- Robust invisible watermarking / C2PA-style provenance metadata doesn't fool a model, but it
  gives you a durable, legally-relevant signal ("this image was marked non-consensual for
  training") that survives compression far better than adversarial noise does. Pairing
  "best-effort ML cloaking" with "durable opt-out signal" is a more honest, layered privacy
  story than ML cloaking alone. Worth at least discussing even if you don't implement it.

---

## B. Shaping this into a strong portfolio project

The single biggest thing that makes this stand out: **be the person who honestly measures
whether cloaking works, instead of the person who claims to have solved it.** This space is
littered with tools that oversell robustness; a project whose headline result is a rigorous,
reproducible robustness evaluation (including where the method *fails*) reads as senior-level
judgment, not just "I implemented a paper."

### Suggested shape
1. **Baseline reproduction** — you already have FGSM. Extend to PGD (iterative, stronger
   than single-step FGSM) as the actual baseline cloaking method.
2. **Break your own baseline** — this is the differentiator. Show empirically that a plain
   PGD cloak:
   - loses effectiveness after JPEG re-compression / resize,
   - loses effectiveness against a purification step,
   - doesn't transfer from the model it was optimized against to a different one.
   Quantify all three with real numbers (verification accuracy before/after, embedding
   cosine-distance shift, etc.).
3. **Add one or two of the "A" improvements above** (ensemble optimization + EOT is the
   highest-leverage, most implementable pair) and re-run the *same* robustness evaluation to
   show a measured improvement — not just "before/after eyeball comparison" but the same
   metrics from step 2, delta'd.
4. **Build a small benchmark harness**, not just a one-off script: a fixed set of test faces,
   a fixed set of "attacker" models (some seen during optimization, some held out to test
   transfer), a fixed set of post-processing steps (compress/resize/purify), and a report
   generator. This harness is arguably the most portfolio-worthy artifact on its own — "I
   built an evaluation framework for photo-cloaking robustness" is a distinct, reusable
   contribution independent of any specific cloaking method.
5. **Ship something usable** — a small CLI or Gradio/Streamlit demo that takes a photo and
   outputs a cloaked version, with the epsilon/method as options and a plain-language
   disclaimer about what threat model it does and doesn't cover. Productization signals matter
   for a portfolio even on a research-flavored project.
6. **Write it up like a mini technical report**: threat model → related work (Fawkes, LowKey,
   Glaze, PhotoGuard, Nightshade, Anti-DreamBooth/MetaCloak) → method → evaluation → honest
   limitations → what you'd do with more time. This is what turns "I ran some scripts" into
   "I can scope, execute, and communicate a research-adjacent engineering project."

### Optional stretch directions (pick at most one, if time allows)
- Extend the same harness to the **generative fine-tuning threat model** (does a LoRA trained
  on your cloaked photos actually fail to reproduce your likeness?) — this is the most
  "currently relevant" framing and would differentiate the project from most Fawkes-era clones.
- A short **certified-radius** experiment via randomized smoothing on a toy model, just to show
  you understand the difference between empirical and certified robustness, even if the main
  tool stays empirical.

Face embedding models (for transferability testing — some used during optimization, some held
  out):
  - FaceNet (Inception-ResNet, facenet-pytorch)
  - ArcFace (InsightFace)
  - CosFace / SphereFace
  - dlib's ResNet face recognizer
  - A CLIP vision encoder (for retrieval/scraping-style similarity search, not verification)
  - A commercial-style black-box stand-in if you want an "unseen model" test (e.g. a different
    open checkpoint you never touch during optimization)

  Purification / preprocessing defenses (the thing that actually kills naive cloaks):
  - DiffPure (diffusion-based purification)
  - JPEG re-compression at a few quality levels
  - Resize/downsample + upsample round-trip
  - Gaussian blur / median filter
  - Simple autoencoder-based denoising

  Adaptive/retraining attacks (the "Fawkes was broken this way" category):
  - Fine-tune or retrain the recognition model on a mix of clean + cloaked images and measure
    recovery
  - Adversarial training of the attacker model against your specific perturbation distribution

  Generative fine-tuning attackers (if you extend to the DreamBooth/LoRA threat model):
  - DreamBooth fine-tune on cloaked photos
  - LoRA fine-tune on cloaked photos
  - Textual inversion

  Protection methods to look into

  - FGSM — your existing baseline
  - PGD (iterative, stronger baseline than FGSM)
  - C&W attack adapted as a cloaking method (better imperceptibility/strength tradeoff)
  - Ensemble-optimized perturbation (optimize jointly against multiple embedding models for
    transferability)
  - EOT (Expectation over Transformation) wrapper — combine with any of the above to survive
    compression/resize
  - Fawkes (the actual reference method, for reproduction/comparison)
  - LowKey
  - PhotoGuard (targets diffusion-model inference/editing, not just classifiers)
  - Anti-DreamBooth
  - MetaCloak
  - Mist
  - Glaze (style-cloaking, worth a look even if your focus is faces — related technique)
  - GAN-based de-identification (CIAGAN, DeepPrivacy) — semantic rather than pixel-noise
    approach
  - Randomized-smoothing / certified perturbation variant — for the certified-robustness angle