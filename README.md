# evogait

QWOP-inspired 2D biped + CPU-only evolution training.

![Evolved 12-body QWOP run](media/walk.gif)

Standalone engine: deterministic 2D sim (`sim-core`), native trainer (`train`), browser playback via `wasm`. Exports versioned gait JSON (see `spec/FORMAT.md`) for any renderer to consume.

## Layout

- `sim-core/` — deterministic physics + CPG policy, zero dependencies
- `train/` — native evolution binary, outputs `out/gait-*.json`
- `wasm/` — `cdylib` browser stepping (same core as native)
- `spec/FORMAT.md` — v1 JSON + WASM contract

## Quick start

```sh
cargo test -p evogait-sim-core
cargo run --release -p evogait-train -- --gens 200 --pop 24 --out out
```

