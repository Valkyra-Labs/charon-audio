# Models: artifacts, verification and hosting

**Charon ships no weights.**

Three ONNX files are supported, all derived from
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

The Demucs code is MIT. Its pretrained weights are not covered by that
license: the maintainer wrote in
[facebookresearch/demucs#327](https://github.com/facebookresearch/demucs/issues/327)
that "the model weights are not covered by the MIT license, and are
provided only for scientific purposes". The official weights repository
(`adefossez/HTDemucs` on Hugging Face) carries no license field, and the
model was trained on MUSDB18-HQ, whose license is non-commercial. Third
party re-hosts that label the weights MIT do so on their own
interpretation. This is why Charon does not host converted weights:
you obtain the checkpoint from its official source and convert it
yourself with the export script, and you assess whether your use is
permitted.

## Hosting

The split exports are not hosted, for the license reason above; produce
them with the export script as shown in the README. Should the rights
holder authorise redistribution, the `url` fields in
`models/manifest.json` and the `download_url` entries in `ModelZoo` will
point at hosted files verified against the hashes above.
