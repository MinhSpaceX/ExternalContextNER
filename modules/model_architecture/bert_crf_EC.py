from transformers import BertModel

import copy
from transformers.models.bert.modeling_bert import BertIntermediate,BertOutput,BertSelfOutput,BertPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
from torchcrf import CRF
import torch
import math
import json
import torch
import torch.nn.functional as F  # For softmax

class BERT_token_classification(BertPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(BERT_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, input_mask_addition, label_ids_addition = None, labels=None):
        features = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
    
        if label_ids_addition is not None:
            main_loss = -self.crf(logits, label_ids_addition, mask=input_mask_addition.byte(), reduction='mean')
            return main_loss
        else:
            pred_tags = self.crf.decode(logits, mask=input_mask_addition.byte())
            return pred_tags


if __name__ == "__main__":
    model = BERT_token_classification.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    print(model)