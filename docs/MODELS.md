# Models: artifacts, verification and hosting

**Charon ships no weights.**

Four ONNX files are supported: three derived from the `htdemucs`
checkpoint of demucs 4.1.0 (drums, bass, other, vocals), and one from
the music branch of TIGER-DnR (below).

| file | contract | provider | SHA-256 | size (bytes) | how to get it |
|---|---|---|---|---|---|
| `htdemucs.onnx` | `Waveform` (STFT inside the graph) | CPU only | `68d0bf16428ef66e692cdff8a9ccf28f1ef3f69440d57e58605a4cc55fcc5e74` | 316,446,953 | download from [StemSplitio/htdemucs-onnx](https://huggingface.co/StemSplitio/htdemucs-onnx) |
| `htdemucs_split.onnx` | `DemucsSplit` | CPU | `6104c3de08607e0898f70835f1be1ff13a85bbea4826bdba80f9cb7b58fe088f` | 185,200,998 | `tools/export/export_htdemucs.py <out> --target cpu` (byte-for-byte reproducible) |
| `htdemucs_split_coreml.onnx` | `DemucsSplit`, tiled 2-D convolutions | CoreML (and CPU, 10% slower) | `782026bd0dbc67e97146271d813f0dc61f5242f80eeefbd6abee5dfe1839607d` | 172,439,242 | `tools/export/export_htdemucs.py <out> --target coreml` (structurally reproducible; bytes differ between runs) |

`models/manifest.json` carries the same table in machine-readable form
and is what `ModelZoo` will read once the files are hosted.

## TIGER-DnR music branch

| file | contract | provider | SHA-256 | size (bytes) | how to get it |
|---|---|---|---|---|---|
| `tiger_music.onnx` | `Spectral` (n_fft 2048, hop 512, 12 s windows, mono per channel), one output `music` | CPU | `bb52541deef92f5f3a62d0e2da916f93dc6caa84ae1e17d3093113c3e8448667` | 25,030,829 | `tools/export/export_tiger.py --repo <TIGER clone> --weights <TIGER-DnR download> --out tiger_music.onnx --dynamo --opset 18` from checkpoint SHA-256 `dd1c696e72f6adea0085ef1af640882a8260519ad666422835e387a5b4abdd2a` |

TIGER-DnR (Xu, Li et al., ICLR 2025) is three separate networks, one per
output (dialogue, effects, music), each predicting three sources and
used for one of them. The export keeps only the music network and its
music source: a third of the full model's work. Byte-for-byte
reproducibility of the export is not verified; the script prints its
parity against PyTorch before writing (95.9-105.8 dB agreement on real
audio).

Licence: the weights on Hugging Face are Apache-2.0 (the code
repository is MIT). The model was trained on Divide and Remaster (DnR),
whose music and effects sources include some non-commercial clips;
neither licence mentions it. Assess your use accordingly.

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
