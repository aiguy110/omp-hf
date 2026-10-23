# omp-hf

An [omp](https://omp.sh) extension that runs small Hugging Face GGUF models locally. Pick one in `/hf`, and the extension downloads the weights and a llama.cpp build, starts `llama-server`, and selects the model. The default list includes [PrismML's Ternary Bonsai 2](https://docs.prismml.com/) models.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/aiguy110/omp-hf/main/install.sh | sh
```

This installs omp if it's missing (using the official `omp.sh/install` script) and copies the extension to `~/.omp/agent/extensions/omp-hf`. Run it again to update.

Then:

```sh
omp                              # run /hf and pick a model
omp --model hf/bonsai-2-4b       # or start on one directly
```

Nothing is downloaded until a model is first used. The first request then fetches:

1. The pinned [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp/releases) build for your platform. This is upstream llama.cpp plus the `PQ2_0`/`PTQ1_0` kernels that Bonsai 2 needs. CUDA is used when `nvidia-smi` is present, ROCm when `rocminfo` is present, Metal on Apple Silicon, and CPU otherwise.
2. The model's GGUF weights, plus its vision projector if it has one, from Hugging Face.

Progress appears in omp's status line. Interrupted downloads pick up where they left off.

## Models

| `/model` id | Model | Download |
|---|---|---|
| `hf/bonsai-2-2b` | Ternary Bonsai 2 2B (PQ2_0 + vision) | 0.9 GB |
| `hf/bonsai-2-4b` | Ternary Bonsai 2 4B (PQ2_0 + vision) | 1.5 GB |
| `hf/bonsai-2-9b` | Ternary Bonsai 2 9B (PQ2_0 + vision) | 3.0 GB |
| `hf/bonsai-2-27b` | Ternary Bonsai 2 27B (PQ2_0 + vision) | 7.8 GB |
| `hf/qwen3-1.7b` | Qwen3 1.7B Q4_K_M | 1.1 GB |
| `hf/qwen3-4b-instruct-2507` | Qwen3 4B Instruct 2507 Q4_K_M | 2.5 GB |
| `hf/gemma-3-4b-it` | Gemma 3 4B IT Q4_K_M | 2.5 GB |

Bonsai models run with PrismML's recommended thinking-mode sampling.

To use any other single-file GGUF repo, run `/hf user/repo` or `/hf user/repo:Q4_K_M`, or choose "Custom Hugging Face repo…" in the picker. Without a quant, `Q4_K_M` is preferred. Custom models are saved to `~/.omp/hf/custom.json`.

## Commands

- `/hf`: open the picker. It shows each model's size and whether it is downloaded or currently serving.
- `/hf <id>` or `/hf user/repo[:QUANT]`: select that model and serve it.
- `/hf status`: show which model is loaded, plus the server PID and log path.
- `/hf stop`: stop `llama-server` and free its memory.

## How it works

- Only one `llama-server` runs at a time, bound to `127.0.0.1:18080`. Selecting a different model restarts it.
- The server is detached from omp. It keeps running after omp exits and is shared by later sessions and subagents. Use `/hf stop` to stop it.
- The `hf` provider uses a custom stream. Before each request, it makes sure the selected model is downloaded and loaded, then passes the request to omp's built-in OpenAI-compatible transport. Switching models with `/model` therefore works too.

Files live in `~/.omp/hf/`: `llama/` (runtime), `models/<repo>/` (weights), `llama-server.log`, and `server.json`.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `OMP_HF_PORT` | `18080` | Server port |
| `OMP_HF_CTX` | per model (32K–64K) | Context size (`-c`) for every model |
| `OMP_HF_BACKEND` | auto | Linux x64: `cpu`, `cuda`, `cuda-12.8`, `cuda-13.3`, `rocm`, `vulkan`. Linux arm64: `vulkan` |
| `OMP_HF_SERVER_ARGS` | | Extra `llama-server` flags, space-separated |
| `OMP_HF_LLAMA_SERVER` | | Use this `llama-server` binary instead of downloading one. For Bonsai 2 `PQ2_0` it must be a PrismML build |
| `OMP_HF_LLAMA_TAG` | `prism-b10787-73bfe6c` | PrismML release tag |
| `OMP_HF_HOME` | `~/.omp/hf` | State directory |
| `HF_TOKEN` | | Hugging Face token, for gated or private repos |

## Notes

- On CPU, the first turn is slow because omp's system prompt is several thousand tokens. Measured on a 22-core laptop CPU with Bonsai 2 2B: about 90 prompt tokens/s and 12 generated tokens/s. Later turns reuse the server's prompt cache.
- Linux and macOS only.

## License

MIT
