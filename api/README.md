---
title: VORTEX-AI Scenario API
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Live crash-scenario generation from the CG-NSDE model
---

Live generation endpoint for the VORTEX-AI research dashboard (https://yash0238.github.io/Vortex-AI/).

- `GET /api/health`: status and available checkpoints
- `POST /api/generate` with `{"n": 200, "seed": 1, "fit": "full"}`: fresh synthetic 60-day windows from the trained
  CG-NSDE (graph-conditioned latent neural SDE), summarised and labelled by the frozen CrisisBench crisis rule.

Research output from a student project (Fr. CRCE, Mumbai). Not investment advice.
