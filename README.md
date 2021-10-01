# SG-MLP
The Official Pytorch Implementation for __Switch Gated Multi-Layer Perceptron(SG-MLP)__.    

SG-MLP, a novel and attentionless architecture for Natural Language Understanding(NLU), achieves decent results in the GLUE benchmark without any help of the Attention Mechanism in both Pre-Training and FineTuning steps. The following repositiory contains demos, pretrained models, and supplementaries necessary for reproducing the results.

<p align="left">
  <img width="446" height="233" src="https://raw.githubusercontent.com/guijinSON/SG-MLP/main/assets/model.png">
</p>

## Masked Language Modeling(MLM) Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17tMXMTt76uw350toP75d-znSRb0DNLYr?usp=sharing)
  
SG-MLP trained on the C4 corpus, learns to predict proper grammer and commonsense knowledge. Refer to the codes below, or the Colab Notebook for a MLM demo ran by our model. (Make sure you have our pretrained model downloaded for the demo) 

```python

from SG-MLP.models.model import build_large_model
from SG-MLP.utils import SGMLP_inference, apply_weight

PATH = '/content/drive/Shareddrives/ICT/weights/model_large_200000.pth'
large_model = build_large_model(output_logits = True)
large_model = apply_weight(large_model,PATH)

SGMLP_inference('A bird has <mask> legs.',large_model)
```


## Team  
* 김승원 - [Seungone Kim](https://github.com/SeungoneKim) 
* 손규진 - [GUIJIN SON](https://github.com/guijinSON)
* 주세준 - [Sejune Joo](https://github.com/joocjun)
* 조우진 - [WOOJIN CHO](https://github.com/WooJin-Cho)
* 채형주 - [Hyungjoo Chae](https://github.com/kyle8581)

