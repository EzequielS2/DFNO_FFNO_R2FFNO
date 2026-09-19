# FNO Variations

PyTorch implementations of three Fourier Neural Operator variants for **1D / 2D / 3D** (`[B, C, *spatial]`):

- **FFNO** — Factorized FNO
- **R2-FFNO** — Reduced-rank Factorized FNO
- **DFNO** — Decomposed FNO

## References

- **FFNO**: [alasdairtran/fourierflow](https://github.com/alasdairtran/fourierflow) — Tran et al., *Factorized Fourier Neural Operators* (ICLR 2023)
- **R2-FFNO**: [Chieh997/R2FFNO](https://github.com/Chieh997/R2FFNO) — Reduced-rank Factorized FNO (ACML 2025)
- **DFNO**: [WenjingYe-HK/Decomposed-FNO](https://github.com/WenjingYe-HK/Decomposed-FNO) — Decomposed Fourier Neural Operator

## Run the 1D training example

```bash
pip install -r requirements.txt
python examples/train_1d.py
# or one model: python examples/train_1d.py --model ffno --epochs 1000
# or python examples\train_1d.py --model all --epochs 1000
```

Plots are written to `examples/outputs/<model>_pred_vs_truth.png`.
