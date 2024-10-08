from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel, XLMRobertaConfig
from torch import nn
from torchcrf import CRF
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
    def __init__(self, config: XLMRobertaConfig, num_labels_: int = 2, auxnum_labels: int = 2):
        super(XLMRoberta_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = XLMRobertaModel(config)
        self.dropout = WordDropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.crf = CRF(num_labels_, batch_first=True)
        self.init_weights()

    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: torch.Tensor, 
        input_mask: torch.Tensor, 
        input_ids2: torch.Tensor, 
        segment_ids2: torch.Tensor, 
        input_mask2: torch.Tensor, 
        labels: torch.Tensor = None, 
        labels2: torch.Tensor = None
    ) -> torch.Tensor:
        
        # Forward pass through first set of inputs
        features = self.roberta(input_ids=input_ids, attention_mask=input_mask)
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            # Forward pass through second set of inputs for auxiliary task
            features2 = self.roberta(input_ids=input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
            sequence_output2 = features2["last_hidden_state"]
            sequence_output2 = self.dropout(sequence_output2)
            logits2 = self.classifier(sequence_output2)

            # Compute CRF loss
            loss1 = -self.crf(logits, labels, mask=input_mask.to(torch.uint8), reduction='mean')
            loss2 = -self.crf(logits2, labels2, mask=input_mask2.to(torch.uint8), reduction='mean')
            main_loss = 0.5 * (loss1 + loss2)
            return main_loss
        else:
            # During inference, decode the CRF output
            pred_tags = self.crf.decode(logits, mask=input_mask.to(torch.uint8))
            return pred_tags
