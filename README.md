# Look It Up: BCH variant (`bch` branch)

This branch holds the codebook-free **BCH variant** from the paper "Look It Up: Multi-bit LLM Watermarking via Codebook Search". It covers only what differs from the main implementation. For the method, installation details, the full results and the citation, see the [README on `master`](https://github.com/jaroslawjanas/llmw-codebook/blob/master/README.md).

## How this branch differs

On `master`, an `n`-bit payload is mapped to a `u`-bit codeword through a secret precomputed table, and extraction searches that table for the nearest codeword. Here the codeword is computed with a BCH (Bose–Chaudhuri–Hocquenghem) error-correcting code and extraction decodes it algebraically, so there is no table to build, store or search.

| | `master` (codebook) | `bch` (this branch) |
|---|---|---|
| Payload to codeword | Lookup in a table of 2<sup>n</sup> codewords | BCH encoding, computed on the fly |
| Extraction | Nearest-neighbour search (Hamming or Levenshtein) | Algebraic BCH decoding |
| Confidence measure | Distance to the nearest codeword | `bit_errors`, the number of bits the decoder corrected |
| Storage | Grows as 2<sup>n</sup> · u | None |
| Payload size in the paper | Up to 24 bits | Up to 32 bits |
| Role of the secret key | Vocabulary split and codebook | Vocabulary split only; the BCH mapping itself is public |
| Robustness | Levenshtein search tolerates deletions and light paraphrasing | Bit substitutions only; an inserted or deleted token shifts every later bit |

Embedding is the same as on `master`: one codeword bit per generated token, green/red list biasing, tokenization-inconsistency correction and multi-GPU generation. Nucleus sampling (`--top-p`) is not available here.

## How the code is chosen

BCH codes have native lengths of 2<sup>m</sup> − 1 bits (63, 127, 255, 511, …). The codec picks its parameters from `--n` and `--u`, for `m` from 3 to 9:

1. If `u` is a native length, the code is used as is.
2. Otherwise the next larger code is *shortened*: unused message positions are fixed to zero and not transmitted. The number of correctable errors `t` is preserved.
3. Only if neither exists is a smaller code padded with zeros, which are not protected.

The settings used in the paper:

| `--n` | `--u` | Code | Corrects `t` | `t / u` |
|---:|---:|---|---:|---:|
| 16 | 127 | BCH(127, 22) | 23 | 18.1% |
| 20 | 127 | BCH(127, 22) | 23 | 18.1% |
| 24 | 127 | BCH(127, 29) | 21 | 16.5% |
| 24 | 200 | BCH(255, 79), shortened by 55 | 27 | 13.5% |
| 32 | 200 | BCH(255, 87), shortened by 55 | 26 | 13.0% |

Prefer a native length. With a 24-bit payload, `--u 127` corrects 21 errors, whereas `--u 128` corrects only 13, because 128 bits have to be cut down from a 255-bit code. The default `--u 100` is not native either (BCH(127, 43), `t = 14`). `--bch-m` forces a particular field degree.

## Setup

Use a separate folder for this branch. `output/` and `cache/` are not tracked, so switching branches inside one folder would mix the runs of both variants, and the scripts here pool every run under `output/` that has the same `n` and `u`.

```bash
git worktree add ../llmw-bch bch      # from an existing clone
cd ../llmw-bch
conda env create -f environment.yml -n llmw-bch
conda activate llmw-bch
```

Without an existing clone, use `git clone -b bch https://github.com/jaroslawjanas/llmw-codebook.git llmw-bch` instead of the first line.

The environment differs from `master`: it pins `transformers` 4.51.3 and PyTorch 2.5.1 with CUDA 11.8, and it does not need Triton. The file declares the same environment name as `master`, which is why the command above passes `-n llmw-bch`. Adjust the PyTorch lines for your GPU as described on `master`.

Gated models need a Hugging Face token in a file named `hf_token` in the folder root, containing the token and nothing else.

## Quick start

A smoke test with a small public model:

```bash
python main.py --model facebook/opt-125m --n 16 --u 127 --n-prompts 5
```

The run prints the selected code, `BCH(n0=127, k=22, t=23)`, and ends with a summary such as `Made 3 successful watermark recoveries out of 3`. Texts that end before `u` tokens are not scored, so the total can be lower than the number of prompts. In a run this short, the reported average BCH decode time is dominated by one-off compilation on the first decode; use `scripts/benchmark_bch_decode.py` for timings.

A setting from the paper, 24 bits in 127 tokens with Llama 2 7B on OpenGen:

```bash
python main.py --model meta-llama/Llama-2-7b-hf \
  --dataset jaroslawjanas/open-generation-data default train prefix \
  --n 24 --u 127 --n-prompts all --context-window 5000
```

### Differences in `main.py`

- `--bch-m` selects the BCH field degree `m`; by default it is chosen automatically.
- `--dataset` takes exactly four arguments naming a Hugging Face dataset (`<name> <subset> <split> <column>`). Local `.json` and `.txt` prompt files are not supported.
- These flags do not exist: `--distance`, `--codeword-generation`, `--top-p`, `--workers`, `--chunk-rows`, `--dedicated-search-gpu`, `--delayed-search`.

The remaining flags are the same as on `master`.

## Output

Each run writes `output/<model>_<dataset>_n<n>_u<u>_<timestamp>/` with `config.json`, `results.txt`, `generations.csv` and `generations.txt`.

- `config.json` has a `bch` block with the selected code: `m`, `n0`, `k`, `t`, `padding` and `shortening`.
- `generations.csv` has the columns `watermark`, `codeword`, `decoded_codeword`, `recovered_watermark`, `bit_errors`, `successful_watermark`, `generated_token_ids`, `decoded_token_ids` and `generated_text`.

`bit_errors` counts the positions where the bits read from the text differ from the codeword of the recovered payload. If at most `t` bits are wrong, the payload is recovered and `bit_errors` is the number of wrong bits. Beyond `t`, the decoder typically returns a different payload with a much larger `bit_errors`, so a threshold on `bit_errors` separates watermarked from non-watermarked text.

## Using the codec directly

```python
from src.bch_codec import BCHCodec

codec = BCHCodec(n=24, u=127)                 # BCH(127, 29), corrects up to 21 bit errors
codeword = codec.encode_int(1234567)          # 127 bits, one per generated token
codeword[[3, 40, 99]] ^= 1                    # three bits read back wrongly
payload, bit_errors = codec.decode_int(codeword)
print(payload, bit_errors)                    # 1234567 3
```

## Analysis scripts

| Script | Purpose |
|---|---|
| `matchrate_at_fpr.py` | Match rate at a target false-positive rate. Decodes `--z` random bit sequences to set a `bit_errors` threshold, then scores every run in `output/` with the given `n` and `u`, grouped by model. |
| `auroc_bch.py` | ROC curve and AUROC for one run: `bit_errors` of successfully recovered texts against random bit sequences. |
| `benchmark_bch_decode.py` | Encoding and decoding time per payload. |
| `attack_robustness.py` | Match rate under bit-level substitution, insertion and deletion applied to the bits read from the text, for every budget from 0 to the maximum. Unlike on `master`, the attacks do not edit the text. |

```bash
python scripts/matchrate_at_fpr.py --n 24 --u 127 --z 10000
python scripts/auroc_bch.py --input-dir output/<run> --z 50000
python scripts/benchmark_bch_decode.py --n 24 --u 127 --z 10000
python scripts/attack_robustness.py --n 24 --u 127 --attack-budget 0.1
```

`<run>` is a run directory created by `main.py`. Each script writes its results to a new directory under `output/`. The first, second and fourth use all CPU cores by default (`--workers`), and each worker takes a few seconds to build its decoder.

## Results and limitations

From the paper, with logit bias `δ = 6` and greedy token selection:

- **Match rate.** 99.1%, 98.6% and 97.6% for 16, 20 and 24 bits in 127 tokens, and 91.1% for 32 bits in 200 tokens (Llama 2 7B on OpenGen).
- **Extraction time.** 2.3 ms per text at `u = 127` and 3.5 ms at `u = 200`, with no dependence on the number of payloads.
- **Low-entropy models.** On instruction-tuned models the match rate for 24 bits drops to 56.5–64.5% at 127 tokens and to 39.6–43.8% at 200 tokens, where the shortened code corrects a smaller share of the bits.
- **Edits.** Decoding compares bits at fixed positions, so an inserted or deleted token breaks it. Use Levenshtein search on `master` when texts may be edited.

The full tables, the comparison with other methods and the citation are in the [README on `master`](https://github.com/jaroslawjanas/llmw-codebook/blob/master/README.md#results).
