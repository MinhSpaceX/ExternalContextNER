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
    def __init__(self, dropout_rate: float = 0.1):
        super(WordDropout, self).__init__()
        assert 0.0 <= dropout_rate < 1.0, '0.0 <= dropout rate < 1.0 must be satisfied!'
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.dropout_rate:
            return inputs

        mask = inputs.new_empty(*inputs.shape[:2], 1, requires_grad=False).bernoulli_(
            1.0 - self.dropout_rate
        )
        mask = mask.expand_as(inputs)
        return inputs * mask

class XLMRoberta_token_classification(XLMRobertaPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(XLMRoberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = XLMRobertaModel(config)
        self.dropout = WordDropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, labels=None, labels2 = None):
        features = self.roberta(input_ids = input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
    
        if labels is not None:
            features2 = self.roberta(input_ids = input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
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

    # def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, labels=None, labels2 = None):
    #     features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
    #     sequence_output = features["last_hidden_state"]
    #     sequence_output = self.dropout(sequence_output)
    #     logits = self.classifier(sequence_output)
    
    #     if labels is not None:
    #         features2 = self.roberta(input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
    #         sequence_output2 = features2["last_hidden_state"]
    #         sequence_output2 = self.dropout(sequence_output2)
    #         logits2 = self.classifier(sequence_output2)
    #         origin_size = logits2.size(1 if logits2.dim() == 3 else 0)
    #         logits_short = logits[..., :origin_size, :]
    #         T=1.0
    #         loss1 = -self.crf(logits, labels, mask=input_mask.byte(), reduction='mean')
    #         loss2 = -self.crf(logits2, labels2, mask=input_mask2.byte(), reduction='mean')
    #         batch_size, max_seq_len, num_classes = logits_short.shape
    #         logits_short = logits_short.detach()
    #         origin_view_log_posterior = self.crf.compute_posterior(logits2, input_mask2.byte())
    #         ext_view_log_posterior = self.crf.compute_posterior(logits_short, input_mask2.byte())
    #         _loss = (
    #             F.kl_div(
    #                 F.log_softmax(origin_view_log_posterior / T, dim=-1),
    #                 F.softmax(ext_view_log_posterior / T, dim=-1),
    #                 reduction='none',
    #             )
    #             * input_mask2.unsqueeze(-1)
    #             * T
    #             * T
    #         )
    #         loss_crf_kl = _loss.sum() / batch_size
    #         main_loss = 0.5 * (loss1 + loss2)  + (1 - 0.5) * loss_crf_kl
    #         return main_loss
    #     else:
    #         pred_tags = self.crf.decode(logits, mask=input_mask.byte())
    #         return pred_tags


if __name__ == "__main__":
    model = XLMRoberta_token_classification.from_pretrained(r'D:\ExternalContextNER\cache\pubmedXLMRoberta')
    print(model)