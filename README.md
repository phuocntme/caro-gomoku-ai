# ♟️ Caro / Gomoku AI

AlphaZero-style Caro (15×15) with a convolutional SE-ResNet policy-value model, vectorized Monte Carlo Tree Search, and a browser-playable AI opponent.

## Play online

[**Open the GitHub Pages demo →**](https://phuocntme.github.io/caro-gomoku-ai/)

The Pages demo supports local **1-vs-1** and **VS AI**. Online multiplayer is disabled there because GitHub Pages only hosts static files; it cannot run the Socket.IO server.

## Run the browser demo locally

Requirements: Python 3 (for a static file server) and an internet connection for ONNX Runtime Web assets loaded from CDN.

```bash
python -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. This serves the same Pages build, including its single-file ONNX model.

## Run the full self-hosted app

The full server build retains **Online multiplayer** as well as 1-vs-1 and VS AI. Requires Node.js 20+ and npm:

```bash
cd server
npm install
npm start
```

Open <http://localhost:8080>. To build and run with Docker, from the repository root:

```bash
docker compose -f server/docker-compose.yml up --build -d
```

The server assets and self-host configuration remain separate from the Pages-only bundle in `docs/`.

## Project overview

- **Network:** fully-convolutional SE-ResNet policy-value model.
- **Search:** vectorized MCTS with PUCT and soft tactical priors.
- **Training:** self-play, opponent pool, checkpoint resume, and ONNX export.
- **Demo:** 15×15 Caro board; ONNX Runtime Web runs the AI in the browser.
- **Multiplayer:** Express + Socket.IO in the self-hosted server only.

## Repository layout

```text
notebooks/Caro.ipynb       Training and self-play pipeline
bench_game.py              Benchmark helper
index.html, script.js,
style.css                  Full self-hosted browser app
checkpoints/               ONNX model for browser inference
server/                    Express + Socket.IO server and Docker setup
docs/                      Static GitHub Pages demo (online mode disabled)
```

## Reproduce training

```bash
pip install -r requirements.txt
jupyter lab notebooks/Caro.ipynb
```

Training runs require a suitable PyTorch environment and any dataset/checkpoints referenced in the notebook. No training metric is claimed here as a reproducible benchmark.

## License and data

No project license is included. Check third-party code, dependency, model, and dataset terms before reuse or redistribution. Do not publish data you are not authorized to share.
