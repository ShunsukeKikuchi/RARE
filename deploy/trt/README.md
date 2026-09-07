# TensorRT experiment scaffolding (not used in the 2026 submission)

Left here because the question "could TRT let us ship more members?" was worth
answering and the groundwork is most of the work. **Nothing in this directory is
part of the shipped container.**

## What we measured, and why TRT was not needed in the end

The 600 s job limit turned out not to bind. Measured on an RTX A4000 (sm_86, the
same compute capability as the platform's A10G, and slower):

| configuration | 384 frames x 30 members x TTA3 |
|---|---|
| batch 64, contended GPU | ~850 s (extrapolated -- this is what triggered the TRT question) |
| **batch 128, dedicated A4000** | **378 s** |
| of which model loading | 81 s (measured on the platform itself) |

So the fix was a batch size the card could actually fill, not a faster runtime.
Before reaching for TRT next time, check `nvidia-smi` utilisation first: the
platform try-out ran at **0% GPU / 0.26% CPU**, which is a loading bottleneck, not
a compute one.

## What we verified about ONNX Runtime

Benchmarked `resnext101_swsl_seg` (our best single member, a pure CNN -- the best
case for ONNX) at batch 32 @384, **with the provider actually confirmed**:

| path | latency | vs PyTorch fp16 | numerics |
|---|---|---|---|
| PyTorch fp16 | 136 ms | 1.00x | -- |
| ONNX Runtime, CUDAExecutionProvider | 909 ms | **0.15x** | max abs diff 0.0051 (clean) |
| ONNX Runtime, TensorrtExecutionProvider | -- | unavailable | TRT libraries not installed |

Two corrections to earlier notes in this repo:

* the ONNX **export is numerically sound** -- an earlier claim that it produced
  NaN and `max|dp| = 1.00` came from a session that was silently on
  CPUExecutionProvider, or from an `ir_version` mismatch;
* ORT's CUDA EP runs the graph in **fp32** (the exporter emits fp32 and the EP does
  not fold to fp16), so that 0.15x compares fp32 against fp16 -- it is not a like
  for like number. Even corrected, the CUDA EP only dispatches cuDNN and has no
  headroom over PyTorch. **TensorRT is the only path worth trying.**

`onnxruntime` 1.21.1 lists `TensorrtExecutionProvider` in
`get_available_providers()`, which means the *build* supports it -- not that the
libraries load. Always assert the provider actually used:

```python
s = ort.InferenceSession(path, providers=["TensorrtExecutionProvider"])
assert "TensorrtExecutionProvider" in s.get_providers()   # else it fell back
```

## If you pick this up

1. **Engines are tied to the GPU architecture and the TRT version.** The platform is
   an A10G (sm_86). dl2 has an RTX A4000, also sm_86 -- build there, not on the
   4090 (sm_89) or the A6000 boxes.
2. **Ship the engine, do not build at startup.** Engine building takes minutes per
   model and would eat the job budget.
3. **Expect the ViT members to be the hard part.** 16 of the 30 wrap every
   `nn.Linear` in a custom `LoRALinear`; SAM2's Hiera, EVA02 and the hand-built Eva
   used for Medical-SAM3 are the likeliest export failures. The 14 seg-aux CNNs are
   the tractable half and would be a sensible first target.
4. **Verify numerics per member against the PyTorch output**, not just that the
   engine runs. Two silent all-NaN incidents this cycle came from precision changes
   that "could not affect anything" -- see the fp16 section in CLAUDE.md.
5. Adding TRT to the container costs ~2-3 GB of image size.

## Files

* `bench_onnx.py` -- the benchmark above; asserts the provider, compares latency and
  numerics against PyTorch for one manifest member.
