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

1. A pinned [PrismML llama.cpp](https://github.com/PrismML-Eng/llama.cpp/releases) build for your platform. This is upstream llama.cpp plus the `PQ2_0`/`PTQ1_0` kernels that Bonsai 2 needs. CUDA is used when `nvidia-smi` is present, ROCm when `rocminfo` is present, Vulkan on Linux x64 machines with an Intel or AMD GPU (including integrated graphics), Metal on Apple Silicon, and CPU otherwise. The Linux x64 Vulkan build comes from this repo's [releases](https://github.com/aiguy110/omp-hf/releases). It is PrismML's source plus [one patch](patches/ggml-vulkan-host-cached.patch), built by [a workflow](.github/workflows/llama-vulkan.yml), and it fixes slow generation on integrated GPUs (see Notes). Set `OMP_HF_BACKEND=cpu` to opt out.
2. The model's GGUF weights, plus its vision projector if it has one, from Hugging Face.

Progress appears in omp's status line. Interrupted downloads pick up where they left off.

## Models

| `/model` id | Model | Download |
|---|---|---|
| `hf/bonsai-2-2b` | Ternary Bonsai 2 2B (PQ2_0 + vision) | 0.9 GB |
| `hf/bonsai-2-4b` | Ternary Bonsai 2 4B (PQ2_0 + vision) | 1.5 GB |
| `hf/bonsai-2-9b` | Ternary Bonsai 2 9B (PQ2_0 + vision) | 3.0 GB |
| `hf/bonsai-2-27b` | Ternary Bonsai 2 27B (PQ2_0 + vision) | 7.8 GB |
| `hf/qwen3.5-2b` | Qwen3.5 2B Q4_K_M (+ vision) | 1.9 GB |
| `hf/qwen3.5-4b` | Qwen3.5 4B Q4_K_M (+ vision) | 3.4 GB |
| `hf/gemma-4-e2b-it` | Gemma 4 E2B IT Q4_0 (+ vision) | 3.4 GB |
| `hf/gemma-4-e4b-it` | Gemma 4 E4B IT Q4_0 (+ vision) | 5.2 GB |

Bonsai and Qwen3.5 models run in thinking mode with their vendors' recommended sampling. Gemma 4 runs without thinking: its thinking switch is a system-prompt token omp does not send. Gemma 4's E2B/E4B files are larger than their names suggest because of per-layer embedding tables.

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

- The first turn is slow without a discrete GPU because omp's system prompt is about 8k tokens. Later turns reuse the server's prompt cache. Bonsai 2 2B on an Intel Core Ultra 7 165H, measured through `llama-server` with an ~8.3k-token prompt:

  | Backend | Prompt tok/s | Generation tok/s |
  |---|---:|---:|
  | CPU | 87 | 11.6 |
  | Vulkan (Arc iGPU), PrismML build | 117–152 | 7.4–8.7 |
  | Vulkan (Arc iGPU), with host-buffer fix | 162 | 19.5 |

  The stock Vulkan build is slow at generation because of how ggml-vulkan allocates pinned host memory, not because of the GPU kernels. It wants host memory that is cached and coherent. Mesa's Intel driver on Meteor Lake offers none (its cached types are non-coherent), so ggml falls back to write-combined memory. The logits output buffer lands there, and sampling reads all ~248K logits from uncached memory every token: about 78 ms per token. `llama-bench` and `llama-completion` hide this because their speed figures leave out sampling time. Falling back to ordinary RAM when no cached and coherent type exists fixes it (a ~15-line change to `ggml-vulkan.cpp`). Upstream llama.cpp has the same code, and large-vocabulary models (Qwen3.5, Bonsai 2, Gemma 4) suffer most.
- Linux and macOS only.

## License

MIT
