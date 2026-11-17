# ♟️ Caro/Gomoku AI — AlphaZero-style Self-Play Engine

An AlphaZero-inspired **Caro (Gomoku, 15×15, Vietnamese rules)** engine: a fully-convolutional **SE-ResNet policy-value network** trained by **self-play with vectorized Monte Carlo Tree Search**, soft tactical priors, and an opponent pool.

## Why this project

Caro is solved-ish territory for minimax at high depth — the interesting question is whether **pure MCTS + learned value** can produce strong play without hand-crafted evaluation, the way AlphaZero did for Go. This project is a full, honest attempt: architecture, self-play curriculum, training ops, evaluation harness, and export to a deployable format.

## Architecture (V2)

- **Board encoding**: 4-channel input planes (my stones, opponent stones, +2 state channels)
- **Network**: fully-convolutional **SE-ResNet** policy-value head (squeeze-and-excitation blocks)
- **Search**: **vectorized MCTS** with PUCT, **soft tactical priors** (4-level heuristics: block open-threes, create open-fours) blended into the root policy
- **Training**: self-play with an **opponent pool** (previous checkpoints) to avoid policy collapse; checkpoint resume from Google Colab or local
- **Export**: `torch.onnx` → ONNX for lightweight inference
- **Evaluation**: `bench_game.py` — head-to-head benchmark harness against heuristics/random

## Repository

```
notebooks/Caro.ipynb               — full pipeline: data gen → training → self-play → eval → ONNX export
bench_game.py                      — CLI benchmark harness (engine vs engine / heuristics)
index.html · script.js · style.css — "Neon Edition" web client: plays the neural net via in-browser
                                     MCTS on the exported ONNX model (onnxruntime-web), plus
                                     socket.io online versus
server/                            — multiplayer relay server (Express + socket.io, Dockerfile)
checkpoints/caro_v2_model_15x15.onnx — trained policy/value model (≈115 KB), loaded by the web client
```

## Quickstart

```bash
pip install -r requirements.txt
jupyter lab notebooks/Caro.ipynb   # reproduce the training pipeline
python bench_game.py --help        # run benchmark games

# play against the AI in the browser
python -m http.server 8000         # open http://localhost:8000

# online multiplayer
cd server && npm install && node server.js
# or: docker compose up
```

## Notes

- The notebook is the single source of truth for the pipeline (research-project style); config cells switch between local and Colab paths.
- Checkpoints/weights are not committed (too large); retrain from scratch in ~a Colab session or reuse the architecture for your own runs.
- Vietnamese Caro rules (15×15, `hậu`/forbid rules per local play conventions) are implemented in the environment module inside the notebook.
