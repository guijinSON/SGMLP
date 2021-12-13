# SG-MLP
The Official Pytorch Implementation for `Switch Gated Multi-Layer Perceptron(SG-MLP)`. Accepted at ACK 2021.

- [SG-MLP: Switch Gated Multi-Layer Perceptron Model for Natural Language Understanding](https://www.koreascience.or.kr/article/CFKO202133649066979.page)

SG-MLP, a novel and attentionless architecture for Natural Language Understanding(NLU), achieves decent results in the GLUE benchmark without any help of the Attention Mechanism in both Pre-Training and FineTuning steps. The following repositiory contains demos, pretrained models, and supplementaries necessary for reproducing the results.

<p align="left">
  <img width="446" height="233" src="https://raw.githubusercontent.com/guijinSON/SG-MLP/main/assets/model.png">
</p>

## Model Config & Pretrained Weights
We trained a total of three models `SG-MLP Small`, `SG-MLP Base` and `SG-MLP Large`.  
The following are the configuration for each models. Pretrained weights for all models are available [here](https://drive.google.com/drive/folders/1FlXtvaC3ZaqOd5zE2lcPiyZ2oKvv1vi1?usp=sharing).

| SG-MLP         | Parameter | Tokenizer              | Corpus                         | Train Steps  |
| -------------- | --------- | ---------------------- | ------------------------------ | ------------ |
| `SG-MLP Small` | ` 67 M`   |  `bert-base-cased`     | `Book Corpus + Wiki`           | `110,000`    |  
| `SG-MLP Base`  | `125 M`   |  `roberta-base`        | `C4`                           | `200,000`    |  
| `SG-MLP Large` | `170 M`   |  `roberta-base`        | `C4`                           | `200,000`    |  

1. Load SG-MLP Base
```python

from SGMLP.models.model import build_base_model
from SGMLP.utils import apply_weight

PATH = '/weights/SGMLP_Base.pth'
base_model = build_base_model()
base_model = apply_weight(base_model,PATH)

```
2. Load SG-MLP Large
```python

from SGMLP.models.model import build_large_model
from SGMLP.utils import apply_weight

PATH = '/weights/SGMLP_Large.pth'
large_model = build_large_model()
large_model = apply_weight(large_model,PATH)

```



## Masked Language Modeling(MLM) Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17tMXMTt76uw350toP75d-znSRb0DNLYr?usp=sharing)
  
SG-MLP trained on the C4 corpus, learns to predict proper grammer and commonsense knowledge. Refer to the codes below, or the Colab Notebook for a MLM demo ran by our model. (Make sure you have our pretrained model downloaded for the demo) 

```python

from SGMLP.models.model import build_large_model
from SGMLP.utils import SGMLP_inference, apply_weight

PATH = '/weights/SGMLP_Large.pth'
large_model = build_large_model(output_logits = True)
large_model = apply_weight(large_model,PATH)

SGMLP_inference('A bird has <mask> legs.',large_model)
```

## Contributors
* 김승원 - [Seungone Kim](https://github.com/SeungoneKim) 
* 손규진 - [GUIJIN SON](https://github.com/guijinSON)
* 주세준 - [SE JUNE JOO](https://github.com/joocjun)
* 조우진 - [WOOJIN CHO](https://github.com/WooJin-Cho)
* 채형주 - [Hyungjoo Chae](https://github.com/kyle8581)

## References

```bibtex
@InProceedings{Zhu_2015_ICCV,
    title = {Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books},
    author = {Zhu, Yukun and Kiros, Ryan and Zemel, Rich and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {December},
    year = {2015}
}
```
```bibtex
@InProceedings{wikitext,
    author={Stephen, Merity and Caiming ,Xiong and James, Bradbury and Richard Socher}
    year=2016
}
```
```bibtex
@article{2019t5,
    author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
    title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
    journal = {arXiv e-prints},
    year = {2019},
    archivePrefix = {arXiv},
    eprint = {1910.10683},
}
```
```bibtex
@InProceedings{wang2019glue,
  title={{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding},
  author={Wang, Alex and Singh, Amanpreet and Michael, Julian and Hill, Felix and Levy, Omer and Bowman, Samuel R.},
  note={In the Proceedings of ICLR.},
  year={2019}
}
```


