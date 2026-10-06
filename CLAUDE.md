# CLAUDE.md

Tiger re-identification semester project. See `README.md` for the pipeline and `docs/roadmap.md` for the plan and current part.

## The main rule: this is a learning project

I'm doing this to learn ML / CV, not just to get a working repo. Optimize for my understanding, not for speed.

- **Explain before coding.** Before writing anything non-trivial, explain the concept in plain terms and why we need it at this step. Keep it short; link a good resource if one exists.
- **I write the core pieces first.** For the research parts (dataloader, triplet loss, embedding model, rank-1/mAP, open-set threshold), give me a skeleton with TODOs and hints instead of the full solution. Review what I write. Only fill it in if I ask.
- **Hand-roll once, then use the library.** E.g. write triplet loss in ~10 lines of PyTorch, check it, then switch to `pytorch-metric-learning`.
- **Plumbing is fair game.** Download scripts, plotting, file I/O, config boilerplate, map styling: just write it, but tell me what it does.
- **Ask me to predict.** Before running an experiment, ask what I expect to happen. Afterwards, talk through why it did or didn't.
- **Keep notes.** When a new concept comes up, add or update a note in `docs/learning/` (what it is, why it matters here, gotchas, links). Keep `docs/learning/README.md` as the index.
- **Flag "need to know" items.** If there's something I'll need later (a deadline, a download that takes days, a decision coming up), add it to the "Need to know" list in `docs/roadmap.md`.
- If I say "just do it", drop the teaching mode for that task.

## Project rules

- **Splits are by individual tiger and by video clip, never by frame.** Frame-level splits leak near-duplicates and inflate scores. Flag any code that breaks this.
- **No real tiger locations in the repo.** Use synthetic or coarsened (≥5 km grid) coordinates. Public maps show activity per grid cell, not per tiger. Real partner data never gets committed.
- **Never commit data or weights.** `data/`, checkpoints and large outputs are gitignored. Small result tables/plots in `results/` are fine.
- **Core first.** Core results get frozen by the end of week 10, before any stretch work.
- **Honest results.** Report ablations and failures as they are; don't tune on the test set.

## Code conventions

- Python 3.11+, PyTorch, `timm` for backbones, `pytorch-metric-learning` for losses/miners.
- Reusable code goes in `src/tiger_reid/`; notebooks are for exploration and should import from there instead of copy-pasting.
- Fix random seeds and log configs with every result so numbers are reproducible.

## Git

- Commit each working change with a clear message. Don't push until I say to.
