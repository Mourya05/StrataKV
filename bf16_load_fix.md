# Safetensors BF16 Load Parallelization

## Diagnosis

The local `model.safetensors` checkpoint is BF16, not F16. The previous loader
optimization parallelized only the F16-to-FP32 path, while the BF16 branch still
ran through the generic conversion loop. The reported production symptom was
approximately 217 seconds of BF16 load/decode time.

## Implementation

`src/io/safetensors.cpp` now applies the same bounded worker architecture to
BF16 conversion:

- tensors with fewer than `1024 * 1024` elements use one direct loop;
- larger tensors use up to eight workers, bounded by
  `hardware_concurrency()`;
- `decode_bf16_range()` reads only its assigned mapped-file range;
- each worker writes only the corresponding range in the pre-allocated FP32
  destination Tensor;
- workers are joined before the converted Tensor enters the WeightMap;
- worker-creation exceptions join already-created workers before rethrowing.

The BF16 conversion preserves the existing little-endian behavior by using
`le16()` and placing the BF16 payload in the upper 16 bits of an FP32 value.
F16 and ordinary F32 decoding behavior remains unchanged.

## Regression coverage

The existing `tests/f16_load_smoke.cpp` now covers both large conversion paths:

- 1,048,576-element F16 tensor, with values `1.0` and `-2.5`;
- 1,048,576-element BF16 tensor, with values `1.0` and `-2.5`;
- first, second, and final values checked;
- output dtype checked as FP32;
- `tensor_decode_ms` printed for both paths.

The test remains registered as `f16_load_smoke` because it covers the original
F16 regression and now also verifies BF16.

## Verification

### Builds and focused tests

Both configurations built successfully:

```powershell
cmake --build build-cpu-check --config Release --target f16_load_smoke phase8_smoke
cmake --build build-cuda-check --config Release --target f16_load_smoke phase8_smoke
```

Focused CTest results:

```text
CPU:   f16_load_smoke + phase8_smoke = 2/2 passed
CUDA:  f16_load_smoke + phase8_smoke = 2/2 passed
```

`git diff --check` passed.

### Focused decode timings

The 1,048,576-element synthetic test reported:

```text
CPU:  F16 13.9498 ms, BF16 3.5649 ms
CUDA: F16  4.2257 ms, BF16 2.2850 ms
```

These values cover conversion only and are not full-model load timings.

### Local model smoke measurement

The local checkpoint was inspected and contains 201 BF16 tensors and no F16
tensors. Running:

```powershell
& .\build-cuda-check\Release\stratakv_cli.exe model.safetensors 0 1
```

after the BF16 change reported:

```text
[TIMING] load_ms = 14469.474800, gen_ms = 0.022700, tokens = 0, pure_decode_tps = 0.000000
[LOAD_BREAKDOWN] mmap_open_ms = 0.1748, header_parse_ms = 0.899, tensor_decode_ms = 4051.16, generator_init_ms = 6104.13
```

This is a post-change measurement on the available local BF16 checkpoint. The
earlier workspace measurements were noisy and varied substantially because of
the large mapped file and host storage/cache state; therefore this report does
not claim a controlled before/after percentage. A fair production comparison
should use identical OS cache state, storage conditions, worker count, and
build, while comparing `LOAD_BREAKDOWN.tensor_decode_ms`.

## Scope and limitations

- The model's BF16 payload remains mapped and is converted into FP32 as required
  by the existing compatibility API.
- The worker cap is eight to match the existing loader policy and avoid creating
  excessive threads on hosts reporting many logical CPUs.
- Existing unrelated phase 5/phase 7 native-BF16 expectation failures remain;
  no tests were changed to hide those contracts.
- No commit was created.
