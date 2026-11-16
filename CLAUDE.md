# CLAUDE.md

Tiger re-identification semester project. See `README.md` for the pipeline and `docs/roadmap.md` for the plan and current phase.

**At the start of every session, read the "START HERE" block at the top of `docs/progress.md`** (current state, next steps, open questions). Update that block before ending a session.

## Working mode: rapid development (since Nov 10, 2026)

Fast, honest results first. Learning is a side effect, not the goal.

- **Claude writes the code**, including the core pieces (splits, losses, model, metrics, open-set threshold). No skeletons-with-TODOs unless I ask.
- **Explain as we go, briefly.** After each step, 2–4 plain sentences: what was built and why it matters. Add a note in `docs/learning/` only for concepts the report will need (index in `docs/learning/README.md`).
- **Test everything.** Each step gets a pytest test or a check script. Run `pytest -q` + the step's check and show the headline numbers before committing.
- **Commit and push after every verified step.** One working change per commit, then `git push` to `origin/main`.
- **Confirm at phase boundaries.** Keep going within a phase; check with me before starting the next one, and whenever a choice affects results (thresholds, split rules, what to drop).
- **Keep the docs current.** `docs/progress.md`: one entry per working day, newest first (Done / Findings / Problems → fixes / To-dos / Next), updated as things happen. `docs/roadmap.md`: tick off steps, keep "Need to know" current. `docs/decisions.md`: log every choice that shapes results (decision, why, alternatives, consequences) so I can explain it later.

## Project rules

- **Splits are by entity and by (pseudo-)clip, never by frame.** Frame-level splits leak near-duplicates and inflate scores. The frozen split file is `splits/splits_v1.csv`; every experiment uses it. Flag any code that breaks this.
- **Open-set entities never touch training.**
- **No real tiger locations in the repo.** Use synthetic or coarsened (≥5 km grid) coordinates. Public maps show activity per grid cell, not per tiger. Real partner data never gets committed.
- **Never commit data or weights.** `data/`, checkpoints and large outputs are gitignored. Small result tables/plots in `results/` and split files in `splits/` are fine.
- **Honest results.** Report ablations and failures as they are; don't tune on the test set (use `val` / `openset_val`).
- **No horizontal-flip augmentation.** It mirrors the stripe pattern into one no real tiger has.

## Code conventions

- Python 3.11+, PyTorch, `timm` for backbones, `pytorch-metric-learning` for losses/miners. Local `.venv` with `pip install -e .`; device is `mps` locally, CUDA on Colab.
- Reusable code goes in `src/tiger_reid/`, runnable entry points in `scripts/`, tests in `tests/`. Notebooks import from `tiger_reid`.
- Fix random seeds and log the config with every result (`results/results.csv`) so numbers are reproducible.
