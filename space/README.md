# space/

"Probe a Fly Brain" demo Space. Two serving modes, both zero live GPU:

- **Static page (default, robust)**: `index.html` + the canned rollout
  cache under `canned/` - five stimulus programs simulated by the real
  `flygpt.sim_backend` LIF on an embedded wiring fixture, cached so the
  Space needs zero live sim calls and zero GPU (T-SHIP-7). Stimulus
  buttons map to annotated sensory populations (sugar GRN, bitter GRN,
  looming LC4/LPLC2, JO afferents, antenna mechanosensors) and decode
  through `flygpt.behavior` (feed/reject/flee/court/groom).
- **ZeroGPU Gradio variant**: `app.py` serves the same canned rollouts
  through a Gradio UI, plus a `flygpt-tiny` forward path when a
  checkpoint is mounted at `FLYGPT_TINY_CKPT`. CPU work outside a
  `spaces.GPU` function burns zero visitor quota.

Regenerate the cache after simulator/config changes:

```bash
python space/build_canned.py --all
```

The headline is validated dynamics prediction, not embodiment theater -
all traces are simulated, not recorded (recorded in
`canned/manifest.json`).
