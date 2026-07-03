# Direct Preference Optimisation for Emotional Depth in Short-Story Generation

**A Length-Normalisation Failure Mode and the Primacy of Preference-Data Alignment**

Final-year dissertation (BSc Computer Science, University of Surrey, 2026). Complete experimental pipeline for fine-tuning **Qwen3-4B** with DPO variants using 4-bit quantised **LoRA**, evaluated across **10 experimental conditions** with an LLM-as-judge pipeline, human annotation, and non-parametric statistical testing.

## Key findings

1. **A previously undocumented length-normalisation failure mode.** Naively dividing sequence log-probabilities by completion length inside the DPO loss inverts the implicit reward scale (chosen rewards of -1.9 to -2.9 despite healthy gradient norms), collapsing the model to 74-92% empty outputs. Training metrics alone looked plausible; the failure was only visible at generation time.

2. **Preference-data alignment dominates hyperparameter tuning.** Rubric-aligned preference pairs recovered baseline quality across all beta values (Cohen's d <= 0.22, not significant), while unfiltered preference data degraded it (d up to -0.99). Beta sweeps could not rescue misaligned data (Kruskal-Wallis beta-insensitivity tests).

3. **Judge validity.** Claude Sonnet and Gemini judges agreed at Spearman rho = 0.81 (with Kendall's tau-b and exact/within-1 agreement rates); judge-vs-human agreement reached rho = 0.90 on a human-annotated subset.

## Pipeline

```
prepare_data.py          Build train/test preference splits (n_train=2500, seed=42)
build_aligned_dataset.py Construct rubric-aligned preference pairs
train_dpo.py             Train: standard DPO / LN-DPO, beta in {0.1, 0.3, 0.5}
generate.py              Sample stories from each condition (base + LoRA adapters)
evaluate.py              LLM-as-judge scoring (Claude Sonnet), 3 rubric dimensions
evaluate_gemini.py       Second judge (Gemini) for inter-rater reliability
human_annotate.py        CLI tool for blind human annotation
calculate_irr.py         Judge agreement: Spearman, Kendall's tau-b, agreement rates
analyse.py               Condition comparisons and summary tables
stats_robustness.py      Welch t-tests, Mann-Whitney U, bootstrap 95% CIs,
                         Kruskal-Wallis, post-hoc power analysis
extract_case_studies.py  Qualitative case-study extraction
figures.py               All dissertation figures
run_experiments.sh       Master script (run per-step inside tmux on the GPU cluster)
```

Evaluation dimensions: emotional flexibility, emotional arc coherence, subtext density (0-3 ordinal scale).

## Technical notes

- **LN-DPO implementation:** `train_dpo.py` subclasses TRL's `DPOTrainer` and overrides the loss computation to length-normalise per-token log-probabilities before computing implicit rewards (`LengthNormalisedDPOTrainer`).
- **Training setup:** 4-bit quantisation (bitsandbytes) + LoRA (PEFT) on a university HPC GPU cluster, run headlessly over SSH/tmux.
- **Statistics:** ordinal-appropriate non-parametric tests throughout; parametric tests reported alongside for robustness; post-hoc power analysis at n=50, alpha=0.05.

## Setup

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt   # torch 2.5.1+cu121, transformers, trl, peft, bitsandbytes
bash scripts/run_experiments.sh   # or run steps individually (recommended)
```

Judge scripts require `ANTHROPIC_API_KEY` / Google API credentials as environment variables. Trained adapters, generated stories, and score files are not committed; the scripts regenerate them.

## Author

Nisaar Bista — [linkedin.com/in/nisaar-bista](https://linkedin.com/in/nisaar-bista)
