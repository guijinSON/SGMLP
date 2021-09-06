from src.pretrain import PreTrainTrainer
from configs import parser

trainer = PretrainTrainer(parser)
trainer.pretrain()
