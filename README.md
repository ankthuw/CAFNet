# CrackVarious: A Multi-Domain Dataset and Crack-Aware Hybrid CNN-Transformer Framework for Robust Crack Segmentation

[![Static Badge](https://img.shields.io/badge/CrackVarious-Dataset?label=Dataset&link=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo)](https://drive.google.com/file/d/16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo) &nbsp;


This repo contain the PyTorch implementation of CrackAwareFusionNet for pixel-wise crack segmentation in civil infrastructure images.

## Update

**2026-01-01**

- Initial public release of CrackAwareFusionNet (CAFNet) code and CrackVarious dataset. The trained model parameters will be published soon.

## 1. Model Architecture

![](/figures/CAFNet.jpg)

- Trained Model: [weight](https://drive.google.com/drive/folders/1NdbbXI3aBjSy5Ipdv07VpPGocdgA1Wl7?usp=sharing) 

## 2. Dataset
- Data organized as:

```text
<CRACKVARIOUS_ROOT>/
  train/
    IMG/
    GT/
  val/
    IMG/
    GT/
  test/
    IMG/
    GT/
```

- Configure the root path in `CrackAwareFusionNet/config.py`:

```python
# CrackAwareFusionNet/config.py
dataset = "./data/CRACKVARIOUS/"  # update to your local path
```

Download link: 
[CrackVarious](https://drive.google.com/file/d/16fOIml_hTxCWjRdqZIZGO7Ci9RWjlcJo/view?usp=sharing) |
[Pavement](https://drive.google.com/file/d/12n5K7Fcb74vT589RJ1d6uVf8JrwX1A7t/view?usp=sharing) |
[Masonry](https://drive.google.com/file/d/1iDTkXXFvd1RGcljVU3JDVLILBnkGkVya/view?usp=sharing) |
[Steel](https://drive.google.com/file/d/1Eq95Cr54m7PPtrKWNbS3b8aLZ6CbG9FU/view?usp=sharing) 

__Please note that the use of our dataset is RESTRICTED to non-commercial research and educational purposes. To download the dataset from the link, please cite as below.__
## 3. Installation

```bash
git clone https://github.com/ankthuw/CAFNet.git
cd CAFNet
pip install -r requirements.txt
```

## 4. Usage
### File Structure
- `CrackAwareFusionNet/model.py` – CAFNet architecture
- `CrackAwareFusionNet/trainer.py` – training loop
- `CrackAwareFusionNet/test.py` – testing / evaluation
- `CrackAwareFusionNet/dataloader.py`, `dataset.py` – data loading utilities
- `CrackAwareFusionNet/config.py` – global configuration
- `CrackAwareFusionNet/utils.py` – helper functions

### Train

From the repo root:

```bash
python -m CrackAwareFusionNet.trainer
# or
python CrackAwareFusionNet/trainer.py
```

Main training configs (epochs, batch size, learning rate, etc.) are defined in `CrackAwareFusionNet/config.py`.

### Test / Evaluation

```bash
python CrackAwareFusionNet/test.py
```

Metrics and evaluation code are in `CrackAwareFusionNet/metric.py`.

## 5. Result
### Result on our dataset

|     Model      | **mIoU (%)** |
|----------------|-------------:|
| UNet       | 67.39        |
| SegFormer | 61.99        |
| HrSegNet-32| 52.72        |
| HrSegNet-48| 59.92        |
| Hybrid-Segmentor | 67.10  |
| CAFNet (Proposed) | **69.41** |

### Comparisions with State-of-the-art
![](/figures/SOTA.jpg)

## 6. Citation

If you use this code or the CrackVarious dataset in your research, please cite our paper (BibTeX will be added after publication).
```
```
If you have any questions, please contact `blathu22@fit.hcmus.edu.vn` or `hmdang22@fit.hcmus.edu.vn` without hesitation.
