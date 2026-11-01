# Paper-to-code mapping

## Research source

*Construction of Financial Risk Assessment Model Based on Text Mining and LLM Architecture*, supplied manuscript by Zhenglin Li, Yiming Zhou, Yuanzhou Wei, and Xinjin Li. Publication venue and final bibliographic metadata were not provided with the implementation inputs.

The manuscript motivates the architecture and evaluation dimensions. The code, generated data, tests, charts, and reported measurements here are a new implementation. The source document, author contact details, manuscript figures, and experimental tables are not redistributed. This repository does not assert authorship of the paper or replication of its published numbers.

| Paper component | Implementation | Scope |
| :--- | :--- | :--- |
| Five-layer architecture | data/evidence/model/CLI/governance modules | Implemented as a local Python package |
| Time decay and source quality | exp(-0.005 * age_days) * quality | Explicit source weights |
| Entity alignment and cosine retrieval | Exact entity IDs plus TF-IDF cosine | No learned entity resolver or neural embedding model |
| Knowledge graph | Entity-document-event JSON graph | Mentions/provenance edges, not inferred causality |
| Event extraction and sentiment | Dictionary events, simple negation, lexical sentiment and entropy | No CRF or pretrained encoder training |
| RAG risk signal | Local Ollama extraction from retrieved evidence | Implemented; live LLM run not included in reported offline metrics |
| Multi-feature fusion | Cost-weighted logistic classifier over financial, narrative and retrieval features; optional LLM covariate | Learned final model, separate from the LLM score |
| Temperature calibration | Positive temperature optimized for log loss | Measured on a held-out calibration subset |
| Time-based training | Purged chronological train/validation/calibration/test split | One fixed extrapolation experiment; not a multi-window rolling backtest |
| Integrated gradients | Exact sigmoid-linear path attribution | On transformed fusion inputs and uncalibrated probability |
| Uncertainty and abstention | Split-conformal label sets, empirical coverage and ambiguous-set abstention | Not individual PD confidence intervals; no temporal coverage guarantee |
| Drift and slicing | DTI-shift PSI, stress metrics and income-quartile slice | Does not certify fairness or robustness |
| Human feedback and deployment | Evidence audit export and batch scoring | Feedback UI, automatic retraining, streaming ingestion and staged releases are future work |
| FinBen/LendingClub and pretrained baselines | Synthetic records; structured LR and TF-IDF text baselines | No redistribution of FinBen/LendingClub; no FinBERT or XGBoost result claimed |

## Statistical clarification

The manuscript's residual-interval construction is not used to claim a distribution-free interval for the underlying default probability. This implementation uses label nonconformity scores 1 - p(observed label) and finite-sample quantile ranks. The usual split-conformal marginal coverage statement requires exchangeability; chronological distribution shift can violate it, so observed test coverage is reported explicitly.

Temperature fitting and conformal quantile estimation use separate sequential subsets. Cost-sensitive fitting can distort probability calibration; temperature scaling is not guaranteed to correct a shifted intercept or improve out-of-time ECE.

## References for implementation choices

- [TF-IDF and text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Temperature scaling, Guo et al.](https://arxiv.org/abs/1706.04599)
- [Integrated gradients, Sundararajan et al.](https://arxiv.org/abs/1703.01365)
- [Conformal prediction introduction, Angelopoulos and Bates](https://arxiv.org/abs/2107.07511)
- [Ollama API](https://docs.ollama.com/api)
