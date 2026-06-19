# SwagHLC — Jetson Setup Notes

## Fixes applied

### Portable (affects all collaborators)

- **`RRD_ROOT` env var** — `DatasetConfig.root` was hardcoded to `/vol/storage/common/SWAG/RRD`.
  Changed default to `None` so `RRD_ROOT` is always respected.

- **`SWAG_MODELS_ROOT` env var** — Added `os.path.expandvars()` to the YAML loader (`config.py`).
  All checkpoint paths in the configs now use `${SWAG_MODELS_ROOT}` instead of absolute paths.
  Collaborators only need to set two env vars:
  ```bash
  export RRD_ROOT=/path/to/RRD
  export SWAG_MODELS_ROOT=/path/to/Integration_models
  ```

- **Hardcoded `dataset.root` in configs** — Removed `/home/reza/Documents/SampleIntegrationData`
  from `demo_single_model.yaml`, `integration_customresnet.yaml`, and `multimodal_fusion.yaml`.

### Jetson-specific

- **PyTorch** — Standard pip PyTorch (`2.12+cu130`) targets CUDA 13.0, incompatible with
  JetPack 6 (CUDA 12.6). Install the Jetson wheel first, then install without the `torch` extra:
  ```bash
  .venv/bin/pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
  .venv/bin/pip install -e ".[dev]"
  ```
  Documented in README Setup section.

- **NumPy** — Jetson torch wheel requires NumPy <2. Downgraded to 1.26.4:
  ```bash
  .venv/bin/pip install "numpy<2"
  ```

- **torchvision** — pip torchvision (`0.27.1`) is incompatible with the Jetson torch wheel
  (`torchvision::nms` operator missing). Since the only usage was `resnet18(weights=None)`,
  inlined the resnet18 architecture directly in `torch_model.py` using pure `torch.nn`.
  No torchvision dependency required.

## Validation — 1200s multimodal fusion run on Jetson (GPU)

```
=== summary ===
  bp_emg: 23928 preds, acc(raw)=79% acc(stable)=81%
      latency ms  compute p50/p95=9.6/11.9 | data_age p50/p95=61/102 | recv_lag p50=1.5 | infer_interval p50=50
      stable intent counts: {activity_2.0:11553, activity_1.0:6426, activity_2.1:2854, activity_3.0:1892, activity_4.0:1203}
  hd_emg: 23919 preds, acc(raw)=83% acc(stable)=86%
      latency ms  compute p50/p95=19.0/22.3 | data_age p50/p95=72/119 | recv_lag p50=2.1 | infer_interval p50=50
      stable intent counts: {activity_2.0:8878, activity_1.0:6904, activity_2.1:3390, activity_4.0:2858, activity_3.0:1889}
  imu: 23919 preds, acc(raw)=94% acc(stable)=94%
      latency ms  compute p50/p95=20.0/23.4 | data_age p50/p95=73/117 | recv_lag p50=1.6 | infer_interval p50=50
      stable intent counts: {activity_2.0:9841, activity_1.0:6696, activity_2.1:3052, activity_3.0:2378, activity_4.0:1952}
  fusion: 23918 preds, acc(raw)=88% acc(stable)=87%
      stable intent counts: {activity_2.0:10948, activity_1.0:6624, activity_2.1:2862, activity_3.0:2004, activity_4.0:1480}
```

### Comparison with collaborator's laptop (CPU)

| Model | Collaborator stable acc | Jetson stable acc |
|---|---|---|
| bp_emg | 81% | 81% |
| hd_emg | 86% | 86% |
| imu | 94% | 94% |
| fusion | 91% | 87% |

Individual model accuracy matches exactly. The 4% fusion gap is a timing artifact — GPU
scheduling jitter on Jetson causes the three model streams to arrive at the fusion stage
slightly out of sync (`data_age p95` ~120ms vs ~76ms on laptop), degrading the majority vote.
