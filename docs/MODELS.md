# Models: artifacts, verification and hosting

charon ships no weights. Three ONNX files are supported, all derived from
the `htdemucs` checkpoint of demucs 4.1.0 (drums, bass, other, vocals).

| file | contract | provider | SHA-256 | size (bytes) | how to get it |
|---|---|---|---|---|---|
| `htdemucs.onnx` | `Waveform` (STFT inside the graph) | CPU only | `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74` | 316,446,953 | download from [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx) |
| `htdemucs_split.onnx` | `DemucsSplit` | CPU | `6104c3de08607e0898f70835f1be1ff13a85bbea4826bdba80f9cb7b58fe088f` | 185,200,998 | `tools/export/export_htdemucs.py <out> --target cpu` (byte-for-byte reproducible) |
| `htdemucs_split_coreml.onnx` | `DemucsSplit`, tiled 2-D convolutions | CoreML (and CPU, 10% slower) | `782026bd0dbc67e97146271d813f0dc61f5242f80eeefbd6abee5dfe1839607d` | 172,439,242 | `tools/export/export_htdemucs.py <out> --target coreml` (structurally reproducible; bytes differ between runs) |

`models/manifest.json` carries the same table in machine-readable form
and is what `ModelZoo` will read once the files are hosted.

## Verify a file

```bash
shasum -a 256 -c models/SHA256SUMS
```

For a re-exported CoreML-target file, whose bytes differ between export
runs, verify structurally instead: same node count (2055), initializer
count (757), op sequence and weights; `tools/export/export_htdemucs.py`
prints the parity against PyTorch (must be below 1e-3, measured 6.6e-5)
before it writes anything.

## License of the weights

The Demucs code is MIT. Its pretrained weights carry no license
statement from the authors (the Hugging Face card of the in-graph export
says MIT on the re-hoster's interpretation). Hosting the split exports
redistributes derived weights; decide whether that is acceptable for
the hosting account before uploading, and say in the model card that
the weights come from `facebookresearch/demucs` (Rouard, Massa,
Défossez, ICASSP 2023) and were converted, not trained.

## Hosting process (to be done by the account owner)

The split exports are not hosted yet. Steps, once the license question
is settled:

1. Create a model repository, for example
   `https://huggingface.co/<org>/charon-htdemucs-onnx`, with a model card
   that states the source checkpoint, the export script commit, the
   demucs/torch versions (4.1.0 / 2.14.0), the hashes above, the
   contract of each file, and the license position.
2. Upload the two files and `SHA256SUMS`:

   ```bash
   pip install huggingface_hub
   ```

   ```bash
   huggingface-cli upload <org>/charon-htdemucs-onnx ~/Music/Charon/models/htdemucs_split.onnx htdemucs_split.onnx
   ```

   ```bash
   huggingface-cli upload <org>/charon-htdemucs-onnx ~/Music/Charon/models/htdemucs_split_coreml.onnx htdemucs_split_coreml.onnx
   ```

   The local copies with verified hashes are in `~/Music/Charon/models/`
   on the development machine.
3. Fill the `url` fields in `models/manifest.json` and the
   `download_url` of the `htdemucs-split*` entries in `src/model_zoo.rs`.
4. Add the split-model regression test to the weekly CI job
   (`.github/workflows/ci.yml`, `real-model` job): download by URL,
   check the hash, run
   `CHARON_HTDEMUCS_SPLIT_MODEL=... cargo test --release --features coreml --test htdemucs -- --ignored`
   on the macOS runner.
5. Re-run `tools/parity/compare.py` on the downloaded files against
   `ref_torch` to confirm the hosted bytes are the measured ones.

A GitHub release asset works the same way (files up to 2 GB); use the
release URL in the manifest.
