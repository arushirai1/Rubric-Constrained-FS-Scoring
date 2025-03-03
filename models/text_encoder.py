import os
import torch

class TextEncoder:
    def __init__(self, model_type, aggregation_method = 'avg'):
        # os.environ['TRANSFORMERS_CACHE'] = '/PATH TO EXTERNAL HUGGINGFACE CACHE/huggingface/'
        self.sentence_bert = False
        self.model_type = model_type
        if model_type == 'clip':
            clip_model_name = "openai/clip-vit-large-patch14"
            if aggregation_method == 'embed':
                from transformers import CLIPModel, CLIPProcessor
                self.model =  CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  # .to(4)
                self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            else:
                from transformers import CLIPTextModel, CLIPTokenizer
                self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
                self.model = CLIPTextModel.from_pretrained(clip_model_name)
        elif model_type == 'bert':
            from transformers import BertModel, BertTokenizer
            # BERT Encoder
            bert_model_name = "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.model = BertModel.from_pretrained(bert_model_name)

        elif model_type == 'roberta':
            from transformers import  RobertaModel, RobertaTokenizer
            roberta_model_name = "roberta-base"
            self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
            self.model = RobertaModel.from_pretrained(roberta_model_name)

        else:
            # sentence bert
            from sentence_transformers import SentenceTransformer
            self.sentence_bert = True # getting embeddings is slightly different
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.aggregation_method = aggregation_method
    def get_sentence_bert_embeddings(self, text):
        return self.model.encode(text)

    def get_clip_model_embeddings(self, text, normalized):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)  # , truncation=True)
        with torch.no_grad():

            text_embeds = self.model.get_text_features(**inputs)
            if normalized:
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds
    def get_embeddings(self, text):
        if self.model_type == 'clip' and self.aggregation_method == 'embed':
            return self.get_clip_model_embeddings(text, normalized=False)[0]
        elif self.sentence_bert:
            return get_sentence_bert_embeddings(text)
        inputs = self.tokenizer(text, return_tensors="pt")
        # print(inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state

        if self.aggregation_method == 'CLS':
            embeddings = embeddings[0][0] # CLS token representation
        else:
            embeddings = embeddings[0, 1:-1, :].mean(0)

        return embeddings