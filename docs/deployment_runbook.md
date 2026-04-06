# GENESIS Two-Machine Deployment Runbook

**Machines**: Dell OptiPlex 7090 (controller) + HP Omen 45L (inference)
**Network**: Same LAN, HTTP between machines
**OS assumed**: Windows on HP Omen, Linux or Windows on OptiPlex (adjust commands accordingly)

---

## Phase 1 — HP Omen 45L (Inference Worker)

### 1.1 Install Ollama

**Windows:**
Download and run the installer from https://ollama.com/download/windows

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:
```bash
ollama --version
```

### 1.2 Pull the model

```bash
ollama pull qwen2.5:32b-instruct-q4_K_M
```

This is ~20 GB. It will load fully into the RTX 5090's 32 GB VRAM.

Verify it loaded:
```bash
ollama list
# Should show: qwen2.5:32b-instruct-q4_K_M
```

Quick sanity check:
```bash
ollama run qwen2.5:32b-instruct-q4_K_M "Reply with exactly: GENESIS READY"
```

### 1.3 Make Ollama reachable on the LAN

By default Ollama only listens on `127.0.0.1:11434`. You must bind it to `0.0.0.0`.

**Windows:**
Set a system environment variable and restart Ollama:
```
setx OLLAMA_HOST "0.0.0.0:11434"
```
Then restart the Ollama app (quit from system tray, relaunch).

**Linux:**
```bash
sudo systemctl edit ollama
```
Add:
```
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```
Then:
```bash
sudo systemctl restart ollama
```

### 1.4 Find the Omen's LAN IP

**Windows:**
```
ipconfig
```
Look for the IPv4 address on your LAN adapter (e.g. `192.168.1.XXX`).

**Linux:**
```bash
ip addr show | grep "inet "
```

Write this IP down. You'll use it on the OptiPlex. Example: `192.168.1.42`

### 1.5 Verify Ollama is listening on LAN

From the Omen itself:
```bash
curl http://localhost:11434/api/tags
```
Should return JSON with your model listed.

---

## Phase 2 — Dell OptiPlex 7090 (Controller)

### 2.1 Install Python 3.12

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip git
```

**macOS (if running macOS on the OptiPlex):**
```bash
brew install python@3.12 git
```

**Windows:**
Download Python 3.12 from https://www.python.org/downloads/ — check "Add to PATH" during install.

Verify:
```bash
python3.12 --version
# Python 3.12.x
```

### 2.2 Clone the repository

```bash
git clone https://github.com/afish2003/GENESIS.git
cd GENESIS
```

### 2.3 Create virtual environment and install

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install --upgrade pip
pip install -e ".[dev]"
```

`sentence-transformers` will pull PyTorch — expect ~2 GB download on first install.

### 2.4 Create .env

```bash
cp .env.example .env
```

Edit `.env` and set the Omen's actual IP:
```
OLLAMA_HOST=http://192.168.1.42:11434
```
Replace `192.168.1.42` with the IP you noted in step 1.4. Leave all other values at defaults.

### 2.5 Verify connectivity to the Omen

```bash
curl http://192.168.1.42:11434/api/tags
```

If this fails: check firewall on the Omen, verify `OLLAMA_HOST=0.0.0.0` is set, verify both machines are on the same subnet.

Then test from Python:
```bash
python -c "
import asyncio
from controller.inference.ollama_backend import OllamaBackend
async def test():
    b = OllamaBackend(host='http://192.168.1.42:11434')
    healthy = await b.health_check()
    print(f'Ollama reachable: {healthy}')
    if healthy:
        from controller.inference.backend import Message
        result = await b.complete([Message(role='user', content='Say GENESIS READY')])
        print(f'Response: {result.content[:100]}')
        print(f'Inference time: {result.total_duration_ms}ms')
    await b.close()
asyncio.run(test())
"
```

Expected output: `Ollama reachable: True` and a response in ~2-5 seconds.

### 2.6 Run tests

```bash
pytest tests/ -v
```

All tests should pass. The retrieval test uses BM25 only (no embedding model needed). The scenario test reads the actual YAML files.

### 2.7 Run a smoke test (3 cycles)

```bash
python scripts/init_run.py --run-id SMOKE_001 --condition BASELINE --cycles 3
python -m controller.main --run-id SMOKE_001 --condition BASELINE --cycles 3
```

This will:
1. Initialize the world from template
2. Run 3 full 14-phase cycles against the Omen's Ollama
3. Produce logs in `research_logs/SMOKE_001/`

Expected runtime: ~5-15 minutes for 3 cycles (depends on model speed).

Verify output:
```bash
ls research_logs/SMOKE_001/
# Should contain: config.json, prompts/, transcripts.jsonl, evaluations.jsonl, etc.

wc -l research_logs/SMOKE_001/*.jsonl
# Should show lines in each log file

python scripts/analyze_run.py --run-id SMOKE_001
# Should print a summary of the 3-cycle run
```

---

## Troubleshooting — Likely First Failures

| Symptom | Cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect` to Omen | Ollama not bound to 0.0.0.0, or firewall blocking port 11434 | Verify `OLLAMA_HOST=0.0.0.0:11434` env var on Omen. Open port 11434 in Windows Firewall or ufw. |
| `ConnectionError: Failed to connect to Ollama after 4 attempts` | Omen went to sleep, Ollama crashed, or wrong IP in .env | Wake Omen, check `ollama list` on Omen, verify IP in .env matches `ipconfig` output |
| `pip install` fails on `sentence-transformers` | Missing C compiler or torch build issue | Install build-essential (`apt install build-essential`) or use `--prefer-binary` flag |
| `ValueError: Failed to parse response as X after 3 attempts` | Model returned malformed JSON | Check `research_logs/SMOKE_001/notable_events.jsonl` for the error. Usually means the model needs a clearer prompt or the temperature is too high for structured output. Try lowering `TEMPERATURE_STRUCTURED` to 0.2 in .env |
| `ModuleNotFoundError` | Not in the venv | Run `source .venv/bin/activate` before any python command |
| Ollama OOM / model not loaded | Model too large for VRAM | Verify `ollama list` shows the model. Check `nvidia-smi` on Omen. qwen2.5:32b-q4_K_M needs ~20 GB VRAM — fits in RTX 5090's 32 GB |
| Tests fail with `No module named 'rank_bm25'` | Deps not installed in active venv | `pip install -e ".[dev]"` inside the activated venv |
| Slow inference (>60s per call) | Model running on CPU instead of GPU | On Omen: `nvidia-smi` should show Ollama using the GPU. If not, reinstall Ollama with CUDA support |
| `FileNotFoundError: World template directory not found` | Running from wrong directory | Always run from the GENESIS repo root: `cd /path/to/GENESIS` |
