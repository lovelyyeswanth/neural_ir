class TextLSTM(nn.Module):
    def __init__(self, model_name_or_dir, n_class, n_hidden):
        super(TextLSTM, self).__init__()

        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir)
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        self.loss = nn.CrossEntropyLoss()

    def encode(self, input_ids, attention_mask, **kwargs):
        # with torch.no_grad():
        outputs = self.model(input_ids, attention_mask, **kwargs)
        # zero-out logits of all padded tokens
        logits = outputs.logits
        logits[~attention_mask.bool()] = 0
        # apply softmax to get probabilities
        probs = nn.functional.softmax(logits, dim=-1)
        # feed the probabilities to the lstm
        input = probs.transpose(0, 1)  # input : [n_step, batch_size, n_class]
        hidden_state = torch.zeros(1, len(input), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(input), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        # take the last output as the encoded vector
        encoded = outputs[-1]  # [batch_size, n_hidden]
        return encoded

    def score_pairs(self, queries, docs):
        # encode queries and docs
        q_vectors = self.encode(queries['input_ids'], queries['attention_mask'])
        d_vectors = self.encode(docs['input_ids'], docs['attention_mask'])
        # calculate scores
        scores = torch.bmm(q_vectors.unsqueeze(1), d_vectors.unsqueeze(2)).flatten()
        return scores

    def forward(self, queries, pos_docs, neg_docs):
        # calculate scores
        pos_scores = self.score_pairs(queries, pos_docs)
        neg_scores = self.score_pairs(queries, neg_docs)
        # calculate the contrastive loss
        targets = torch.zeros_like(pos_scores).type(torch.LongTensor).to(pos_scores.device)
        loss = self.loss(torch.cat((pos_scores.reshape(-1, 1), neg_scores.reshape(-1, 1)), dim=1), targets)
        return loss, pos_scores, neg_scores

    def save_pretrained(self, model_name_or_dir, state_dict=None):
        self.model.save_pretrained(model_name_or_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir, n_class, n_hidden):
        return cls(model_name_or_dir, n_class, n_hidden)
