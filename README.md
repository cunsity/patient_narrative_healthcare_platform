# Patient narratives as text-derived evaluation signals in online healthcare

Reproducibility materials for **“Natural language processing of patient narratives in digital healthcare: Validation of text-derived evaluation signals in a Chinese online healthcare platform.”**

## Study and access

The study analyzes 30,005 de-identified patient reviews associated with 5,156 physicians from a Chinese online healthcare platform. The preferred complete-case regression sample contains 28,433 reviews. The original platform-level records and review narratives are **not** included in this public repository because their redistribution is restricted and may raise privacy concerns.

The public repository provides analysis code, variable definitions, and verification instructions. Running the complete empirical analysis requires access to the restricted model-ready dataset under appropriate authorization.

## Main measures

- `rating`: 1–5 patient rating.
- `sentiment_continuous`: **categorical polarity code** −1 (negative), 0 (neutral), +1 (positive); the inherited column name does *not* indicate a continuous probability or score.
- `sentiment_score_raw`: model confidence, distinct from the polarity code.
- `complaint_present`: binary complaint-language indicator derived from review text and tags.
- `gratitude_present`: binary appreciation-language indicator derived from review text and tags.
- `informativeness`: exploratory composite proxy, not a validated ground-truth measure.
- `doctor_id`: de-identified physician grouping variable for cluster-robust standard errors.

## Reproduce final supplementary robustness checks

1. Install Python dependencies from `requirements.txt`.
2. Obtain authorized private copies of `paper3_model_ready_part1.csv` through `paper3_model_ready_part5.csv` and place them in a local directory **outside the Git repository**.
3. Run:

```bash
python verify_final_models.py --data-dir /path/to/private/csv_parts --output-dir verification_results
```

The script reproduces binary logit (S2), ordered logit (S3), residual diagnostics (S4), and missing-title sensitivity (S5). Ordered logit with grouped departments can take considerable time; for a faster preliminary check add `--skip-ordered`.

The binary outcome is `highest_rating = 1` for a 5-star review, and 0 for ratings 1–4. Ordered logit retains the 1–5 rating scale. The binary and ordered models group departments with fewer than 500 complete-case reviews into `Other`. The preferred OLS includes full department and review-year fixed effects and physician-clustered standard errors. Ordered-logit standard errors follow the notebook's standard maximum-likelihood implementation and are **not** physician-clustered.

## Data integrity note

An intermediate local CSV export contained 1,048,575 rows, of which 1,018,570 had no review identifier and contained only a spurious `informativeness` value. The authorized model-ready dataset was reconstructed by retaining the 30,005 rows with valid unique review IDs. The anomalous export is not part of this repository.

## Submission checklist

- Confirm the study-period description against source timestamps and the submitted manuscript.
- Confirm Supplementary Table S6 describes `sentiment_continuous` as categorical, not continuous.
- Ensure that private review text, raw platform data, coder files, credentials, and patient-level CSVs are not committed.
- Check the final analysis notebook for local absolute paths before publication.
