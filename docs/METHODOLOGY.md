# Methodology

## Financial records and labels

The generator creates one entity and assessment per sample over approximately six years, with debt-to-income ratio, income, principal, delinquencies, rate, and a financial narrative. A disclosed logistic mechanism samples default labels from affordability, payment history, narrative event burden, and evidence-event burden. Outcomes become available after a 90-day maturation interval. This is generated data, not an observed loan cohort or an empirical default-horizon estimate.

Each entity receives four documents: three historical reports and a future report. Documents have publication and availability times; two days of ingestion delay are modeled. Future reports are stored for filtering tests and excluded from the assessment evidence set. No names, author emails, employer records, or real borrower identifiers are present.

## Temporal boundaries

Records are ordered by assessment date and divided at 55%, 70%, and 85%. In training, validation, and calibration, rows are purged if their outcome is unavailable at the next partition's first assessment. The published split retains 917 training, 197 validation, 197 calibration, and 270 test rows.

Retrieval vocabulary is fit exclusively on documents available before the validation period. Document eligibility checks entity ID, publication timestamp and availability timestamp against each assessment. A future vocabulary token is tested explicitly. There is no target label in the retrieval corpus or predictor set.

The calibration period is split sequentially: the first 98 rows fit temperature and the remaining 99 fit conformal quantiles. Each model's numeric imputer, percentile bounds, scaler, narrative TF-IDF, and logistic coefficients fit only training rows. Thresholds use validation outcomes with illustrative false-negative cost 3 and false-positive cost 1. The full fusion architecture is designated in advance; holdout metrics do not select it or tune it.

## Retrieval and feature fusion

Text similarity uses TF-IDF cosine within exact entity matches. Eligible documents receive source quality times exponential age decay. The three highest positive similarity-weight products become evidence. Duplicate normalized content per entity retains earliest availability. Low source quality or old evidence reduces its contribution. This ranking is a transparent lexical implementation, not a semantic embedding or entity-linking benchmark.

Narrative event, sentiment, uncertainty and entropy signals are concatenated with financial inputs, raw narrative TF-IDF, and weighted retrieved event/sentiment/source features. Cost-sensitive logistic regression uses positive-class weight 3. The optional LLM signal is appended only when matching records pass input-hash and evidence-ID checks; no simulated LLM score is substituted in offline mode.

## Calibration, explanations and uncertainty

Temperature scales a binary logit with a positive scalar optimized on calibration log loss. It does not necessarily remove class-weighting bias or improve ECE. Attribution is exact integrated gradients for the uncalibrated sigmoid-linear aggregator in transformed feature space using a zero baseline; scaled numeric zero means the training mean and text zero is an empty vector. The audit displays the eight largest absolute contributions, so their truncated sum need not equal the full probability difference.

Conformal label sets include class y when 1-p(y) is at most the calibration quantile. Sets with zero or two labels trigger abstention, as does missing evidence. The usual nominal marginal coverage interpretation requires exchangeability. Empirical temporal-test coverage, average set size and abstention are reported; none is a guarantee about individual default probabilities.

## Evaluation

Report F1, MCC, average precision, ROC-AUC, Brier score, equal-width ten-bin ECE, and confusion matrices. Model-only scoring latency excludes retrieval, calibration fitting, training, LLM execution and file I/O; text/evidence feature extraction latency is reported separately. No GPU or real-time throughput claim is made.

Error buckets can overlap and must not be added as disjoint totals. The DTI stress test adds 0.10 to holdout ratios, reuses existing labels, and measures score sensitivity and PSI; it does not simulate causal economic outcomes. Income-quartile metrics are a descriptive slice, not a fair-lending assessment.

## Local LLM interface

The connector sends data only to a local Ollama endpoint. Source documents are labeled as untrusted evidence, and returned JSON must contain a numeric risk score, cited available document IDs, and an explanation. Missing evidence, malformed responses, and invented citations fail validation. These controls do not prove factual reasoning or prompt-injection resistance. Unit tests use mocked responses; live generation requires a separately installed model. Its pretraining data and dates must be considered in temporal evaluation.

The llm-enrich command writes one record per assessment, including model name, prompt version, input hash and evidence IDs. Use the same synthetic seed and row count for training. For scoring a subset, filter the signal file to exactly those sample IDs. A hash checks consistency, not authenticity or tamper resistance.
