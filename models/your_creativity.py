from torch import nn
import torch
from transformers import BertModel

# Modify the MyDenseBiEncoder class to use cross attention
class MyDenseBiEncoder(nn.Module):

    def __init__(self, model_name_or_dir, n_heads) -> None:
        super().__init__()
        # Use a pre-trained model with a cross attention layer
        self.model = BertModel.from_pretrained(model_name_or_dir, add_cross_attention=True)
        self.loss = nn.CrossEntropyLoss()
        # Add a multi head attention layer after the model
        self.attention = torch.nn.MultiheadAttention(n_heads)

    def encode(self, input_ids, attention_mask, **kwargs):
        # Get the query and document inputs from the kwargs
        query_input_ids = kwargs["query_input_ids"]
        query_attention_mask = kwargs["query_attention_mask"]
        # Pass the query and document inputs to the model with cross attention
        outputs = self.model(input_ids, attention_mask=attention_mask, encoder_hidden_states=query_input_ids, encoder_attention_mask=query_attention_mask, **kwargs)
        # Get the last hidden state of the model
        last_hidden_state = outputs.last_hidden_state
        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        last_hidden_state = last_hidden_state.masked_fill(mask == 0, 0.0)
        # Apply the multi head attention layer to the last hidden state
        attention_output = self.attention(last_hidden_state, last_hidden_state, last_hidden_state, mask)
        # Avg over hidden states to get document representation
        sum_attention_output = torch.sum(attention_output, dim=1)
        sum_attention_mask = torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
        mean_attention_output = sum_attention_output / sum_attention_mask.unsqueeze(-1)
        return mean_attention_output

    def score_pairs(self, queries, docs):
        q_vectors = self.encode(queries.input_ids, queries.attention_mask)
        d_vectors = self.encode(docs.input_ids, docs.attention_mask)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        scores = cos(q_vectors, d_vectors)
        return scores

    def forward(self, queries, pos_docs, neg_docs):
        pos_scores = self.score_pairs(queries, pos_docs)
        neg_scores = self.score_pairs(queries, neg_docs)
        loss = self.loss(torch.cat((pos_scores, neg_scores), dim=0), torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), dim=0))
        return loss, pos_scores, neg_scores

    def save_pretrained(self, model_dir, state_dict=None):
        self.model.save_pretrained(model_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir, n_heads):
        return cls(model_name_or_dir, n_heads)