from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig
)
from torch import nn

# adapted from https://github.com/facebookresearch/DPR/blob/main/dpr/models/hf_models.py

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("bert-base-uncased", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                         attention_mask=attention_mask,return_dict=False)
        pooled_output = sequence_output[:, 0, :]

        return sequence_output, pooled_output, hidden_states
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size

class QueryClassifier(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, class_num):
        super(QueryClassifier, self).__init__()
        # note here we only have question encoder
        self.question_model = HFBertEncoder.init_encoder()
        self.classifier = nn.Linear(self.question_model.config.hidden_size, class_num, bias=False)


    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        return pooled_output
    def forward(self, query_ids, attention_mask_q, return_hidden_emb=False):
        q_embs = self.query_emb(query_ids, attention_mask_q)
        if return_hidden_emb:
            return q_embs
        logits = self.classifier(q_embs)
        return logits




