# VORTEX-AI (CG-NSDE) — BE Project Seminar Presentation

## Complete Slide Deck & Speaker Notes

**Duration:** 18-20 minutes | **Audience:** Engineering Faculty + Laymen | **Date:** 29 August 2026

---

## SLIDE 1: Title Slide (30 seconds)

### Visual
- Project title: "VORTEX-AI: Contrastive Graph-Neural Stochastic Differential Equations for Financial Scenario Generation"
- Your name, department, university
- Guide name
- NIFTY-50 stock exchange visual background

### Speaker Notes
"Good morning everyone. Today I'll be presenting VORTEX-AI — a framework that combines Graph Neural Networks with Stochastic Differential Equations to generate synthetic financial market scenarios. Specifically, we model NIFTY-50 — India's top 50 companies — to understand and predict how financial crises spread across sectors. Let me start with why this matters."

---

## SLIDE 2: Motivation — Why Study Financial Crises? (2 minutes)

### Visual
- Photo: 2008 financial crisis news headline / COVID-19 market crash chart
- Simple diagram: Company A → Company B → Company C (contagion effect)
- Statistic: "During COVID-19, NIFTY-50 fell 38% in March 2020"

### Speaker Notes
"Imagine you're watching 50 of India's biggest companies — Reliance, TCS, HDFC Bank. On normal days, their stock prices move independently. But during a crisis — like 2008 or COVID-19 — they all fall together. This is called the 'contagion effect' — like a cold spreading through an office.

The problem is: we don't have enough historical crisis data to train prediction models. India has had only 3-4 major crashes in 15 years. A machine learning model needs thousands of examples to learn properly.

Our solution: Teach a computer to *generate* realistic synthetic crisis scenarios — like a flight simulator for financial markets — so we can stress-test portfolios and prepare for the next crash."

---

## SLIDE 3: Real-World Analogy (1.5 minutes)

### Visual
- Left: Flight simulator cockpit
- Right: Stock market dashboard
- Middle: "=" symbol connecting them
- Bottom: "Synthetic scenarios → Better preparedness"

### Speaker Notes
"Think of what pilots do. They don't learn to fly during real emergencies — they use flight simulators that generate thousands of realistic crash scenarios. We're building the same thing for financial markets.

Our system learns the 'physics' of how stocks move together during normal and crisis periods. Then it can generate millions of realistic market scenarios — including extreme crashes that haven't happened yet — so regulators and portfolio managers can prepare."

---

## SLIDE 4: Research Gap (2 minutes)

### Visual
- Three columns:
  - **Old Approach 1:** Static correlation matrix — "One snapshot, misses dynamics"
  - **Old Approach 2:** Single-stock models — "Ignores connections between stocks"
  - **Old Approach 3:** Simple GANs — "Generate data but no structural understanding"
- Bottom arrow pointing to our approach: "Dynamic graphs + Neural SDE + Contrastive learning"

### Speaker Notes
"After surveying 25+ papers, we identified three key gaps:

**Gap 1:** Most studies use a single static correlation matrix — one snapshot of how stocks relate. But these relationships change! During crises, correlations spike.

**Gap 2:** Many models treat each stock independently — like studying 50 patients without considering they might infect each other.

**Gap 3:** Existing generative models — like GANs — can create synthetic data, but they don't understand the *structure* of the market. They don't know that TCS and Infosys are in the same sector.

Our work combines three innovations: dynamic graph attention, neural stochastic differential equations, and contrastive learning — to address all three gaps simultaneously."

---

## SLIDE 5: Literature Survey Summary (2 minutes)

### Visual
- Table with 4 columns: **Paper** | **Year** | **Approach** | **Limitation We Address**
- 5-6 key papers listed:
  - Diebold & Yilmaz (2014) — Spillover index — "Static, no generation"
  - Veličković et al. (2018) — GAT paper — "No financial application"
  - Kidger et al. (2021) — Neural SDE — "No graph conditioning"
  - TimeGAN (2020) — Temporal GAN — "No structural awareness"
  - Our work (2026) — CG-NSDE — "Dynamic + Structural + Generative"

### Speaker Notes
"Our literature survey spans four research threads:

First, **financial connectedness** — Diebold & Yilmaz's spillover index measures how shocks spread, but it's static. We make it dynamic.

Second, **graph attention networks** — Veličković et al. invented GAT for social networks. We adapt it for financial markets.

Third, **neural SDEs** — Kidger et al. showed how to learn differential equations from data. We add graph conditioning to their framework.

Fourth, **time-series generation** — TimeGAN generates realistic sequences. We add structural awareness through graphs.

Our contribution: the first framework to combine all four for financial scenario generation on Indian markets."

---

## SLIDE 6: Problem Statement (1 minute)

### Visual
- Boxed statement in center:
  > "How can we generate realistic synthetic financial scenarios that capture both the temporal dynamics and the structural relationships between assets during crisis regimes, given limited historical crisis data?"
- Three bullet points below:
  - Limited crisis data for model training
  - Need for structural + temporal modeling
  - Requirement for stress-testing capabilities

### Speaker Notes
"Our problem statement: How can we generate realistic synthetic financial scenarios that capture both temporal dynamics and structural relationships between assets during crisis regimes?

This matters because regulators need to stress-test banks, portfolio managers need to estimate tail risks, and current tools fail when faced with unprecedented crises — exactly when they're needed most."

---

## SLIDE 7: Objectives (1.5 minutes)

### Visual
- Numbered list with icons:
  1. 📊 Build reproducible 60-day NIFTY-50 pipeline
  2. 🏷️ Label crisis regimes using composite scoring
  3. 🔗 Construct dynamic Pearson + Granger graphs
  4. 🧠 Train DynamicGAT + Neural SDE model
  5. 📈 Generate synthetic scenarios for stress-testing
  6. ✅ Evaluate using statistical + discriminative metrics

### Speaker Notes
"Our six objectives map directly to our methodology:

One — build a reproducible data pipeline for NIFTY-50, the benchmark index of India's National Stock Exchange.

Two — automatically label crisis periods using a composite score combining volatility, drawdowns, and circuit breaker proximity.

Three — construct dynamic graphs that capture how stock relationships change over time.

Four — train our novel CG-NSDE model combining graph attention with neural stochastic differential equations.

Five — generate synthetic market scenarios for stress-testing.

Six — evaluate using statistical metrics and a discriminative score — can a classifier tell real from synthetic data?"

---

## SLIDE 8: SDG Mapping (1 minute)

### Visual
- Three SDG icons:
  - **SDG 8** (Decent Work) — "Economic resilience through early crisis detection"
  - **SDG 9** (Innovation) — "Explainable AI for financial infrastructure"
  - **SDG 16** (Strong Institutions) — "Transparent risk analysis for regulators"
- Connection lines from project to each SDG

### Speaker Notes
"Our project maps to three UN Sustainable Development Goals:

**SDG 8 — Decent Work and Economic Growth:** Earlier detection of systemic financial stress supports more resilient economic activity, protecting jobs.

**SDG 9 — Industry, Innovation and Infrastructure:** We develop an explainable graph-based research prototype for financial infrastructure.

**SDG 16 — Peace, Justice and Strong Institutions:** Reproducible evaluation and transparent risk indicators support accountable decision-making in financial regulation."

---

## SLIDE 9: System Architecture — High-Level (2 minutes)

### Visual
- Left-to-right flowchart:
  - [Yahoo Finance] → [Preprocessing] → [60-day Windows] → [Dynamic Graphs] → [CG-NSDE Model] → [Synthetic Scenarios]
- Icons for each block
- Bottom: "Blocks 0-2 complete | Blocks 3-5 upcoming"

### Speaker Notes
"Let me walk you through our architecture:

**Input:** We download 15 years of NIFTY-50 prices from Yahoo Finance — 50 stocks, 3700 trading days.

**Preprocessing:** We convert prices to log-returns, label crisis days, and create 60-day sliding windows — each window is a 3-month snapshot of all 50 stocks.

**Dynamic Graphs:** For each window, we build a 'who-moves-with-whom' map using Pearson correlation and Granger causality — a statistical test for whether one stock predicts another.

**CG-NSDE Model:** Our core innovation — a Graph Attention Network encodes the structure, a Neural SDE generates temporal paths, and a contrastive head separates normal from crisis in the latent space.

**Output:** Synthetic market scenarios that match the statistical properties of real markets.

Today I'll present our progress through Block 2 — the Dynamic GAT Encoder."

---

## SLIDE 10: Data Pipeline — Visual Explanation (1.5 minutes)

### Visual
- Top: Stock price chart → Log-returns chart (jagged line)
- Middle: Sliding window animation (60-day box moving across time)
- Bottom: Crisis labeling — green (normal) and red (crisis) shaded regions

### Speaker Notes
"Let me explain our data pipeline with a visual:

**Step 1:** We take raw stock prices and convert them to log-returns — the percentage change each day. This makes different stocks comparable.

**Step 2:** We label each day as crisis or normal. Our crisis score combines three signals: high volatility (prices swinging wildly), drawdown fraction (many stocks falling together), and circuit breaker proximity (stocks hitting exchange limits).

**Step 3:** We create sliding windows of 60 days — about 3 months. Each window captures the recent history of all 50 stocks.

The result: 3641 windows, each containing 60 days × 50 stocks, with a crisis label for each window."

---

## SLIDE 11: Dynamic Graphs — Explain Like I'm 5 (2 minutes)

### Visual
- Left: 50 dots (stocks) with some connected by lines
- Right: Same dots with MORE lines (crisis state)
- Labels: "Normal: sparse connections" vs "Crisis: tight coupling"
- Small text: "Pearson = concurrent | Granger = predictive"

### Speaker Notes
"Now, what do we mean by 'graph'? Imagine each stock as a dot. We draw a line between two dots if their prices move together.

**In normal times:** The graph is sparse — stocks move independently. Maybe 15% of possible connections.

**In crisis times:** The graph becomes dense — everything moves together. Correlations jump to 40-50%.

We use two types of connections:
- **Pearson correlation:** Do two stocks move together right now? (concurrent)
- **Granger causality:** Does one stock *predict* another? (lead-lag)

Our optimized Granger test runs in 1.3 seconds per window — down from minutes using traditional methods."

---

## SLIDE 12: What is Graph Attention? (2 minutes)

### Visual
- Center: One stock (TCS) surrounded by neighbors
- Arrows with weights: thick arrow to Infosys (0.8), thin arrow to Reliance (0.2)
- Formula: `attention_weight = how much should I listen to this neighbor?`
- Caption: "Like a student in a study group — learn more from similar peers"

### Speaker Notes
"Here's the key insight of Graph Attention Networks:

When predicting what TCS will do, we shouldn't treat all stocks equally. We should 'pay attention' to the most relevant neighbors. TCS is an IT company, so it should listen more to Infosys (another IT company) than to Reliance (oil & gas).

The network learns these attention weights automatically. During training, it discovers sector structures — IT stocks attend to IT stocks, banks attend to banks.

Our implementation uses torch_geometric's GATConv layer with 4 attention heads in the first layer — meaning it learns 4 different 'perspectives' on which neighbors matter."

---

## SLIDE 13: DynamicGATEncoder Architecture (2 minutes)

### Visual
- Layer diagram:
  - Input: (Batch, Time, Nodes, Features=6)
  - ↓ GATConv(6→64, heads=4) + LayerNorm + ELU
  - ↓ GATConv(256→64, heads=1) + LayerNorm + ELU
  - ↓ Attention extraction → Learned adjacency
  - Output: (Batch, Time, Nodes, Hidden=64) + Learned adjacency
- Side: "Graph pooling → (Batch, 64) for SDE conditioning"

### Speaker Notes
"Here's our DynamicGATEncoder architecture:

**Input:** Each stock gets 6 features — its return, absolute return, volume, sector ID, circuit band limit, and distance to that band.

**First GAT layer:** Maps 6 features to 64 hidden dimensions using 4 attention heads. Each head learns different neighbor importance.

**Second GAT layer:** Maps 256 dimensions (4 heads × 64) down to 64 output dimensions.

**Key innovation:** We extract the attention weights from the second layer to form a *learned* adjacency matrix — the network's own understanding of market structure.

**Output:** 64-dimensional embeddings for each stock at each timestep, plus the learned graph structure.

**Graph pooling:** We average across stocks to get a single 64-dimensional 'graph embedding' — this will condition our Neural SDE in Phase 3."

---

## SLIDE 14: 20% Implementation Evidence (2 minutes)

### Visual
- Checklist with green checkmarks:
  - ✅ Full data pipeline (download → returns → regimes → windows)
  - ✅ 6-dim node features with volume integration
  - ✅ Optimized Granger causality (parallel processing)
  - ✅ DynamicGATEncoder (torch_geometric, 3D+4D)
  - ✅ Graph consistency loss verification
  - ✅ Attention non-uniformity validation
  - ✅ Gradient flow verification
- Bottom: "GitHub: github.com/yash0238/Vortex-AI"

### Speaker Notes
"Let me show you our 20% implementation progress:

**Complete:**
- Full data pipeline — download, preprocess, label, window — producing 3641 windows
- 6-dimensional node features including volume data
- Optimized Granger causality with parallel processing — 100x speedup
- DynamicGATEncoder with torch_geometric — supports both 3D and 4D inputs
- Graph consistency loss — measuring how well learned graphs match empirical ones
- Attention visualization — confirming non-uniform, meaningful attention weights
- Gradient flow verification — all parameters receive gradients during training

All code is available on GitHub with detailed commit history. We've also created a comprehensive Jupyter notebook with 39 cells documenting every step."

---

## SLIDE 15: Results & Visualizations (2 minutes)

### Visual
- 4 images in grid:
  - Top-left: Cumulative returns with crisis shading
  - Top-right: Attention weight heatmap (non-uniform)
  - Bottom-left: Graph consistency loss comparison (normal vs crisis)
  - Bottom-right: Per-layer gradient norms

### Speaker Notes
"Here are four key results:

**First:** Our cumulative returns chart shows NIFTY-50's 15-year journey with crisis periods shaded in red — 2008, 2020, and others.

**Second:** Our attention weight heatmap — notice it's NOT uniform. The network pays more attention to certain stock pairs, learning meaningful structure.

**Third:** Graph consistency loss is higher in crisis regimes — meaning the learned adjacency diverges more from the empirical one during stress. This is expected and informative.

**Fourth:** Gradient norms confirm healthy training dynamics across all layers — no vanishing or exploding gradients."

---

## SLIDE 16: What's Next? Phases 3-6 (1.5 minutes)

### Visual
- Timeline:
  - ✅ Phase 0-2: Data + GAT (COMPLETE)
  - ⏳ Phase 3: Neural SDE (torchsde integration)
  - ⏳ Phase 4: Latent Encoder + Decoder
  - ⏳ Phase 5: Contrastive Regime Head
  - ⏳ Phase 6: Circuit Filter + Loss Assembly
  - ⏳ Phase 7+: Training, Evaluation, Paper

### Speaker Notes
"Here's our roadmap forward:

**Phase 3:** Implement the Neural SDE using torchsde — this is the generative engine that will produce synthetic market paths. The GAT embeddings will condition the SDE's drift and diffusion.

**Phase 4:** Add a latent encoder (GRU-based) and decoder that maps SDE outputs back to realistic returns.

**Phase 5:** Implement the contrastive regime head using Supervised Contrastive Loss — this ensures synthetic crises look different from synthetic normal periods.

**Phase 6:** Add the NSE circuit filter — a hard constraint that clips returns at exchange limits, plus a soft penalty during training.

**Phases 7+:** Full training, evaluation against TimeGAN/QuantGAN baselines, and paper writing for ICAIF 2027."

---

## SLIDE 17: Individual Contribution & Ethics (1 minute)

### Visual
- Three columns:
  - **Solo Work:** All implementation, architecture design, experiments
  - **Reproducibility:** Version control, cached artifacts, documented pipeline
  - **Ethics:** Research prototype — NOT trading advice, survivorship bias acknowledged

### Speaker Notes
"A note on individual contribution and ethics:

This is a solo project — I designed the architecture, implemented all components, and conducted all experiments.

For reproducibility, I use GitHub version control, cache all intermediate artifacts, and document every pipeline step.

Ethically, I want to emphasize: this is a research prototype, NOT a trading recommendation. We acknowledge limitations like survivorship bias — we only study stocks that survived 15 years — and the distinction between correlation and causation."

---

## SLIDE 18: Conclusion (1 minute)

### Visual
- Three key takeaways:
  1. "First framework combining Dynamic GAT + Neural SDE for Indian markets"
  2. "20% implementation complete with validated components"
  3. "Open-source, reproducible, SDG-aligned research"
- Thank you message

### Speaker Notes
"To conclude:

**First:** We present the first framework combining Dynamic Graph Attention with Neural Stochastic Differential Equations for financial scenario generation on Indian markets.

**Second:** Our 20% implementation is complete and validated — data pipeline, node features, optimized graphs, and GAT encoder all working correctly.

**Third:** This is open-source, reproducible research aligned with UN Sustainable Development Goals 8, 9, and 16.

Thank you. I'm happy to take questions."

---

## SLIDE 19: Q&A / Backup Slides

### Visual
- "Thank you! Questions?"
- Contact info, GitHub link

### Backup slides (if needed):
- Mathematical formulation of Neural SDE
- Detailed attention mechanism equations
- Full literature comparison table
- Architecture diagram with tensor dimensions

---

## Anticipated Q&A Preparation

**Q: Why not just use a simple GAN?**
A: Simple GANs generate data without understanding market structure. Our approach combines structural awareness (GAT), temporal dynamics (SDE), and regime separation (contrastive) — giving us interpretable, structurally-aware generation.

**Q: What about transaction costs and market impact?**
A: That's Phase 6 — the NSE circuit filter models exchange-level constraints. Transaction costs are a future extension.

**Q: How do you validate synthetic data quality?**
A: Three ways: (1) Statistical fidelity — do synthetic returns match real kurtosis, ACF, correlation? (2) Discriminative score — can a classifier tell real from synthetic? (3) Domain metrics — do crises show higher correlation than normal periods?

**Q: Why NIFTY-50 specifically?**
A: India is an emerging market with fewer crashes than developed markets — making data scarcity a real problem. If our approach works here, it's valuable for regulators globally.

**Q: What's novel compared to TimeGAN?**
A: TimeGAN models temporal patterns but ignores cross-sectional structure. Our model captures BOTH: temporal (via SDE) and structural (via GAT), with regime-aware generation (via contrastive loss).

---

## Presentation Timing Summary

| Section | Slides | Time |
|---|---|---|
| Motivation & Analogy | 2-3 | 3.5 min |
| Research Gap & Literature | 4-5 | 4 min |
| Problem & Objectives | 6-7 | 2.5 min |
| SDG Mapping | 8 | 1 min |
| Architecture | 9 | 2 min |
| Technical Deep-Dive | 10-13 | 7 min |
| Implementation & Results | 14-15 | 4 min |
| Future Work | 16 | 1.5 min |
| Ethics & Conclusion | 17-18 | 2 min |
| **Total** | **18** | **~20 min** |

---

## Key Speaking Tips

1. **For laymen:** Use the flight simulator analogy (Slide 3) — it clicks immediately
2. **For faculty:** Emphasize the novelty — first combination of GAT + SDE + Contrastive for finance
3. **When stuck:** Point to the architecture diagram — it's self-explanatory
4. **Confidence builders:** 
   - "We've verified gradient flow through all layers"
   - "Attention weights are provably non-uniform"
   - "All code is publicly available on GitHub"
5. **Handle tough questions:** Acknowledge limitations honestly — "That's a limitation we address in Phase 6"
