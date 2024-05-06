from transformers import XLMRobertaModel

import copy
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaIntermediate,XLMRobertaOutput,XLMRobertaSelfOutput,XLMRobertaPreTrainedModel
from transformers import XLMRobertaConfig
from torch import nn as nn
from torchcrf import CRF
import torch
import math
import json
import torch
import torch.nn.functional as F  # For softmax
class WordDropout(nn.Module):
    """Word-level Dropout module

    During training, randomly zeroes some of the elements of the input tensor at word level
    with probability `dropout_rate` using samples from a Bernoulli distribution.

    Args:
        dropout_rate (float): dropout rate for each word
    """

    def __init__(self, dropout_rate: float = 0.1):
        super(WordDropout, self).__init__()
        assert 0.0 <= dropout_rate < 1.0, '0.0 <= dropout rate < 1.0 must be satisfied!'
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Dropout the input tensor at word level

        Args:
            inputs (torch.Tensor): input tensor

        Returns:
            outputs (torch.Tensor): output tensor of the same shape as input
        """
        if not self.training or not self.dropout_rate:
            return inputs

        mask = inputs.new_empty(*inputs.shape[:2], 1, requires_grad=False).bernoulli_(
            1.0 - self.dropout_rate
        )
        mask = mask.expand_as(inputs)
        return inputs * mask

class Roberta_token_classification(XLMRobertaPreTrainedModel):
    """Coupled Cross-Modal Attention Roberta model for token-level classification with a softmax layer on top.
    """
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(Roberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = XLMRobertaModel(config)
        self.dropout = WordDropout(config.hidden_dropout_prob)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(num_labels_, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)

        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, labels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            main_loss = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            return main_loss
        else:
            pred_tags = self.crf.decode(logits, mask=input_mask.byte())
            return pred_tags

if __name__ == "__main__":
    model = Roberta_token_classification.from_pretrained('microsoft/BiomedNLP-BiomedRoberta-base-uncased-abstract-fulltext')
    print(model)