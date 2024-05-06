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


class XLMRoberta_token_classification(XLMRobertaPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(XLMRoberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, labels=None, labels2 = None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
    
        if labels is not None:
            features2 = self.roberta(input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
            sequence_output2 = features2["last_hidden_state"]
            sequence_output2 = self.dropout(sequence_output2)
            logits2 = self.classifier(sequence_output2)

            loss1 = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
            loss2 = -self.crf(logits2, labels2, mask=input_mask2.byte(), reduction='mean')
            main_loss = 0.5 * (loss1 + loss2)
            return main_loss
        else:
            pred_tags = self.crf.decode(logits, mask=input_mask.byte())
            return pred_tags


if __name__ == "__main__":
    model = XLMRoberta_token_classification.from_pretrained(r'D:\ExternalContextNER\cache\pubmedXLMRoberta')
    print(model)