# evogait FORMAT v2

Versioned contract between the engine and any renderer (web widget or
otherwise). Renderers must reject `format != 2` loudly, not silently.
(v1 was a 4-joint toy and is retired.)

Coordinates: +x right, +y DOWN, metres, gravity +10 — verbatim from the
`bicrick/qwop-python` Box2D plan. Track top surface at `y = 9.14275`.

## `gait-*.json`

```json
{
  "format": 2,
  "fps": 30,
  "bodies": ["torso", "head", "leftArm", "leftForearm", "leftThigh",
             "leftCalf", "leftFoot", "rightArm", "rightForearm",
             "rightThigh", "rightCalf", "rightFoot"],
  "keys": [3, 2, 0, 3, 1, 7, 3, 2, 0, 3, 7, 0],
  "distance": 0.158,
  "fell": false,
  "frames": [[[x, y, angle] x 12], ...]
}
```

- `format`: integer, must be `2`.
- `fps`: frames per second, `30`.
- `bodies`: fixed 12-name order above; each frame holds 12 ×
  `[x, y, angle]` (angle in radians).
- `keys`: evolved Discrete-9 key loop (`0` none, `1` Q, `2` W, `3` O,
  `4` P, `5` QW, `6` QP, `7` WO, `8` OP), each held `LOOP_HOLD`
  physics steps. Replaying it through `QwopSim` regenerates the run.
- `distance`: torso travel in metres over the sample window.
- `fell`: whether the sample ended in a face-plant.
- `frames`: body states at `fps`, 8 s window (`<= 240` frames).

## `reward.json`

```json
{"history": [[gen, best_score], ...]}
```

`best_score` = distance minus `2.0` when the rollout fell or slumped
(torso below the `FALL_Y` upright cutoff).

## WASM (`evogait-wasm`)

- `new Walker()` / `reset()`
- `step_action(a: 0..8) -> torso_x`, `step_idle() -> torso_x`
- `torso_x()`, `torso_y()`, `fallen()`, `body_count()`
- `bodies(out: Float64Array[36])`, `extents(out: Float64Array[24])`
- Fixed 25 Hz step, `f64` core, no randomness in stepping.
