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



class BertCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertCrossAttention(nn.Module):
    def __init__(self, config):
        super(BertCrossAttention, self).__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output

class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertCrossAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BERT_token_classification(BertPreTrainedModel):
    def __init__(self, config, num_labels_=2, auxnum_labels=2):
        super(BERT_token_classification, self).__init__(config)
        self.num_labels = num_labels_
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(num_labels_, batch_first=True)
        self.text_dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.text_ouput_cl = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifier = nn.Linear(config.hidden_size, num_labels_)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)

        
        self.image_dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.image_output_cl = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()
        self.cross_attention = BertCrossEncoder(config, 1)

    def text_toimage_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (torch.matmul(text_h1[i], image_h1[i]) / (torch.norm(text_h1[i]) * torch.norm(image_h1[i]))) / temp
            )

            down = torch.sum(
                torch.exp((torch.sum(text_h1[i] * image_h1, dim=-1) / (
                            torch.norm(text_h1[i]) * torch.norm(image_h1, dim=1))) / temp), dim=-1)

            loss += -torch.log(up / down)

        return loss

    def image_totext_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (
                        torch.matmul(image_h1[i], text_h1[i]) / (torch.norm(image_h1[i]) * torch.norm(text_h1[i]))
                ) / temp
            )
            down = torch.sum(
                torch.exp((torch.sum(image_h1[i] * text_h1, dim=-1) / (
                            torch.norm(image_h1[i]) * torch.norm(text_h1, dim=1))) / temp), dim=-1)
            loss += -torch.log(up / down)
        return loss

    def total_loss(self,text_h1, image_h1, temp, temp_lamb):
        lamb = temp_lamb
        batch_size = text_h1.shape[0]
        loss = (1 / batch_size) * (
                    lamb * self.text_toimage_loss(text_h1, image_h1, temp) + (1 - lamb) * self.image_totext_loss(text_h1, image_h1, temp))
        # print("total_loss:",loss)
        return loss

    def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, input_mask_addition, label_ids_addition = None, labels=None):
        features = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        features2 = self.bert(input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
        sequence_output = features["last_hidden_state"]
        sequence_output_pooler = features["pooler_output"]
        sequence_output = self.dropout(sequence_output)
        sequence_output2 = features2["last_hidden_state"]
        sequence_output_pooler2 = features2["pooler_output"]
        sequence_output2 = self.dropout(sequence_output2)

        extended_attention_mask = input_mask2.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        cross_encoder = self.cross_attention(sequence_output, sequence_output2, extended_attention_mask)
        cross_output_layer = cross_encoder[-1]

        final_output = torch.cat((sequence_output2, cross_output_layer), dim=-1) # batch_size * seq_len * 2(hidden_size)
        logits = self.classifier(final_output)  # batch_size * seq_len * 13
    
        if label_ids_addition is not None:
            text_output_cl = self.text_ouput_cl(self.relu(self.text_dense_cl(sequence_output_pooler)))
            image_ouput_cl = self.image_output_cl(self.relu(self.image_dense_cl(sequence_output_pooler2)))
            temp = 0.5
            temp_lamb = 0.5
            cl_loss = self.total_loss(text_output_cl, image_ouput_cl, temp, temp_lamb)
            main_loss = -self.crf(logits, label_ids_addition, mask=input_mask_addition.byte(), reduction='mean')
            alpha = 0.3
            # print(f"main_loss: {alpha * main_loss} cl_loss: {(1 - alpha) * cl_loss}")
            loss_plus = alpha * main_loss + (1 - alpha) * cl_loss
            return loss_plus
        else:
            pred_tags = self.crf.decode(logits, mask=input_mask_addition.byte())
            return pred_tags


if __name__ == "__main__":
    model = BERT_token_classification.from_pretrained(r'D:\ExternalContextNER\cache\pubmedbert')
    print(model)


# from transformers import BertModel

# import copy
# from transformers.models.bert.modeling_bert import BertIntermediate,BertOutput,BertSelfOutput,BertPreTrainedModel
# from transformers import BertConfig
# from torch import nn as nn
# from torchcrf import CRF
# import torch
# import math
# import json
# import torch
# import torch.nn.functional as F  # For softmax



# class BertCrossEncoder(nn.Module):
#     def __init__(self, config, layer_num):
#         super(BertCrossEncoder, self).__init__()
#         layer = BertCrossAttentionLayer(config)
#         self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

#     def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
#         all_encoder_layers = []
#         for layer_module in self.layer:
#             s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
#             if output_all_encoded_layers:
#                 all_encoder_layers.append(s1_hidden_states)
#         if not output_all_encoded_layers:
#             all_encoder_layers.append(s1_hidden_states)
#         return all_encoder_layers

# class BertCoAttention(nn.Module):
#     def __init__(self, config):
#         super(BertCoAttention, self).__init__()
#         if config.hidden_size % config.num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (config.hidden_size, config.num_attention_heads))
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
#         mixed_query_layer = self.query(s1_hidden_states)
#         mixed_key_layer = self.key(s2_hidden_states)
#         mixed_value_layer = self.value(s2_hidden_states)



#         query_layer = self.transpose_for_scores(mixed_query_layer)
#         key_layer = self.transpose_for_scores(mixed_key_layer)
#         value_layer = self.transpose_for_scores(mixed_value_layer)



#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
#         attention_scores = attention_scores + s2_attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.Softmax(dim=-1)(attention_scores)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)
#         return context_layer

# class BertCrossAttention(nn.Module):
#     def __init__(self, config):
#         super(BertCrossAttention, self).__init__()
#         self.self = BertCoAttention(config)
#         self.output = BertSelfOutput(config)

#     def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
#         s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
#         attention_output = self.output(s1_cross_output, s1_input_tensor)
#         return attention_output

# class BertCrossAttentionLayer(nn.Module):
#     def __init__(self, config):
#         super(BertCrossAttentionLayer, self).__init__()
#         self.attention = BertCrossAttention(config)
#         self.intermediate = BertIntermediate(config)
#         self.output = BertOutput(config)

#     def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
#         attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
#         intermediate_output = self.intermediate(attention_output)
#         layer_output = self.output(intermediate_output, attention_output)
#         return layer_output

# class BERT_token_classification(BertPreTrainedModel):
#     def __init__(self, config, num_labels_=2, auxnum_labels=2):
#         super(BERT_token_classification, self).__init__(config)
#         self.num_labels = num_labels_
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         # self.classifier = nn.Linear(config.hidden_size, num_labels_)
#         self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)
#         self.crf = CRF(num_labels_, batch_first=True)

#         self.init_weights()
#         self.cross_attention = BertCrossEncoder(config, 1)

#     def forward(self, input_ids, segment_ids, input_mask, input_ids2, segment_ids2, input_mask2, input_mask_addition, label_ids_addition = None, labels=None):
#         features = self.bert(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
#         features2 = self.bert(input_ids2, token_type_ids=segment_ids2, attention_mask=input_mask2)
#         sequence_output = features["last_hidden_state"]
#         sequence_output = self.dropout(sequence_output)
#         sequence_output2 = features2["last_hidden_state"]
#         sequence_output2 = self.dropout(sequence_output2)
#         extended_attention_mask = input_mask.unsqueeze(1).unsqueeze(2)

#         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
#         # masked positions, this operation will create a tensor which is 0.0 for
#         # positions we want to attend and -10000.0 for masked positions.
#         # Since we are adding it to the raw scores before the softmax, this is
#         # effectively the same as removing these entirely.
#         extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
#         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

#         cross_encoder = self.cross_attention(sequence_output, sequence_output2, extended_attention_mask)
#         cross_output_layer = cross_encoder[-1]

#         final_output = torch.cat((sequence_output2, cross_output_layer), dim=-1) # batch_size * seq_len * 2(hidden_size)
#         logits = self.classifier(final_output)  # batch_size * seq_len * 13
    
#         if label_ids_addition is not None:
#             main_loss = -self.crf(logits, label_ids_addition, mask=input_mask_addition.byte(), reduction='mean')
#             return main_loss
#         else:
#             pred_tags = self.crf.decode(logits, mask=input_mask_addition.byte())
#             return pred_tags


# if __name__ == "__main__":
#     model = BERT_token_classification.from_pretrained(r'D:\ExternalContextNER\cache\pubmedbert')
#     print(model)
