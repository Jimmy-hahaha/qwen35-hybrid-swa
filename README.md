# qwen35-hybrid-swa

Sliding-window attention support for **Qwen3.5** (`QWEN35` arch) in llama.cpp, so the
periodic full-attention layers of a hybrid GDN/attention checkpoint get a **bounded KV cache**
instead of one that grows with the context.

Patches are against llama.cpp `aa75ff323`.

## Problem

Qwen3.5 is a 3:1 hybrid: three Gated DeltaNet (linear attention, fixed-size recurrent state)
blocks followed by one gated softmax attention block. The recurrent layers already have
constant memory; only the attention blocks grow.

SWA-converted checkpoints (e.g. `wxqnl030/GDN-SWA-0.8B`, which turns the 6 attention layers
into `sliding_attention` with `sliding_window: 512`) cannot express that in llama.cpp, for two
independent reasons:

1. **Conversion drops the metadata.** `conversion/qwen.py :: Qwen3_5TextModel.set_gguf_parameters()`
   only exports `full_attention_interval`. `sliding_window` / `sliding_window_pattern` are never
   written, and the layer classifier only treats `layer_types[i] == "linear_attention"` as
   recurrent, so `sliding_attention` silently falls through to dense attention.
2. **Runtime ignores SWA for this arch.** `src/models/qwen35.cpp` never calls
   `load_swa_pattern()`, so even a GGUF that does carry the keys loads with `swa_type == NONE`.
   And the graph uses the generic `build_attn_inp_kv()`, which hard-asserts
   `swa_type == NONE` ("Use llama_kv_cache_iswa for SWA"). A hybrid model needs
   `llm_graph_input_mem_hybrid_iswa`, the split base/SWA cache input.

Result without the patch: a 6-layer × `n_ctx` dense KV cache. For `-c 8192` that is 96 MiB, and
it doubles with every context doubling.

## Changes

Three files, +80/−10 lines.

### `conversion/qwen.py` — export the metadata

`Qwen3_5TextModel.set_gguf_parameters()` now writes `qwen35.attention.sliding_window` and
`qwen35.attention.sliding_window_pattern` when the HF config declares
`layer_types[i] == "sliding_attention"`. Stock checkpoints have no `sliding_window` key and are
unaffected.

### `src/models/models.h` — templatize the graph

`llama_model_qwen35::graph` becomes `graph<bool iswa>`, selecting
`llm_graph_input_attn_kv_iswa` vs `llm_graph_input_attn_kv` via `std::conditional_t`, the same
pattern LFM2/LFM3 use for their hybrid+SWA graphs. Adds a `get_rope_params()` override.

### `src/models/qwen35.cpp` — wire it up

- `load_arch_hparams()`: parse `LLM_KV_ATTENTION_SLIDING_WINDOW`, set
  `swa_type = LLAMA_SWA_TYPE_STANDARD`, call `load_swa_pattern(ml, 4)` (period 4, default
  `dense_first = false` → layer indices 3/7/11/... are SWA, matching the Qwen3.5 3:1 layout).
  The per-layer pattern array from the GGUF is authoritative; `n_pattern` is only the fallback.
- `build_arch_graph()`: dispatch to `graph<true>` when `swa_type != NONE`.
- Graph constructor: `build_inp_mem_hybrid_iswa()` vs `build_inp_mem_hybrid()`.
- Explicit instantiations for `graph<true>` / `graph<false>` at the end of the file.

## Two non-obvious gotchas

**1. Rope base gets overwritten.** `llama_model::get_rope_params()` returns
`rope_freq_base_train_swa` (default `10000`) for SWA layers, but Qwen3.5 uses `1e7`. Left alone,
positions past ~10k are garbage. The patch overrides `get_rope_params()` so SWA layers keep the
global base/scale (same treatment as LFM2).

**2. The MTP draft block has no split cache.** `block_count` is 25 for a 24-layer trunk (1 NextN
layer). The MTP block sits outside the hybrid layer filters and cannot use the split base/SWA
cache, so `graph_mtp` now throws a clear error instead of hitting an assert deep in the graph.

## Verified

`Qwen3.5-0.8B` converted from `wxqnl030/GDN-SWA-0.8B`, `--outtype q8_0`, 795 MB.

```
print_info: n_swa                      = 512
print_info: is_swa_any                 = 1
llama_kv_cache: size   =   0.00 MiB (     0 cells, 0 layers)   # non-SWA
llama_kv_cache: SWA    =  12.00 MiB (  1024 cells, 6 layers)    # 512 window, 2 seq slots
llama_memory_recurrent: size = 19.27 MiB                        # GDN state, constant
```

| | before | after |
|---|---|---|
| `n_swa` / `is_swa_any` | 0 / 0 | 512 / 1 |
| non-SWA KV | 96.00 MiB (8192 cells × 6) | 0.00 MiB (0 layers) |
| SWA KV | — | 12.00 MiB (1024 cells × 6) |
| KV at `-c 8192` → `-c 16384` | 96 → 192 MiB | 12.00 → 12.00 MiB |

Total cache is flat from 4K to 256K: 12 MiB (SWA) + 19.27 MiB (recurrent state).

Reproduce with `-v` and compare the `llama_kv_cache` lines at two context sizes. Do not trust the
metadata alone — a wrong graph path can still report `is_swa_any = 1` while allocating a dense
cache, so check that the KV size is byte-identical at `-c 8192` and `-c 16384`.

Prebuilt GGUFs of the model above are in the release assets: `gdn-swa-0.8b-f16.gguf` (1.5 GB)
and `gdn-swa-0.8b-q8_0.gguf` (795 MB). They already carry the sliding-window metadata, so they
only need the runtime patch.

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout aa75ff323
git apply /path/to/patches/0001-conversion-export-sliding-window.patch
git apply /path/to/patches/0002-qwen35-graph-hybrid-swa.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

`qwen35-hybrid-swa.patch` is the same thing combined, for `git apply` in one shot.

## Limitations

- **Tested only on Qwen3.5-0.8B** (24 layers, 6 SWA blocks), CPU build. The 2B/9B/27B hybrid
  checkpoints use the same arch code path but have not been run.
- **MTP + SWA is rejected**, not supported. A checkpoint with `n_layer_nextn > 0` and a sliding
  window raises. Supporting it needs the draft block routed to a dedicated dense cache.
- **Prefill activations still scale.** A constant KV cache is not constant VRAM: an 8K prefill
  chunk still spikes several GB of activations. Use chunked prefill for large `-c`.
- **Quality is the checkpoint's problem, not the patch's.** The attention layers were trained
  dense; truncating them to 512 tokens costs long-range recall. Short-context behaviour is
  unchanged. Long-range precision needs something like Engram (see the 27B full-linear variants).
- The 512 window is per sequence slot; `n_seq_max` > 2 multiplies the SWA cache.

## Upstream

Not submitted yet. The change is self-contained and mirrors the existing LFM2/LFM3 hybrid+SWA
pattern, so it should be a reasonable PR — but the graph templatization touches a large file and
needs a broader test matrix (multiple sizes, CUDA/Metal, `n_seq_max > 1`) first.

## License

MIT, same as llama.cpp. Patch content is derived from llama.cpp source.
