# Ownership

- New model implementation belongs to the user: src/pusht_diffusion/learner/models.py.
- All new training algorithms belong to the user: src/pusht_diffusion/learner/training.py.
- Keep those methods NotImplementedError unless the user explicitly authorizes implementation.
- Never hide model/training solutions in tests, examples, fallback predictors, or diagnostic scripts.
- baselines/act.py is a pre-existing vendored baseline; retain provenance.
- No model training, cloud runs, or modifications to the original mini-wam checkout are authorized by scaffolding work.
- Parent owns docs/architecture* and docs/assets/architecture* and illustration prompts/assets.
- Distinguish mock engineering checks, real smoke checks, and formal trained-policy results.
