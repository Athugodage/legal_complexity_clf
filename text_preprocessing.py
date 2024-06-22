import re
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    
    Doc
)

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)


class Speller():
    def __init__(self,
                 model : str):
        self.model = M2M100ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = M2M100Tokenizer.from_pretrained(model, src_lang="ru", tgt_lang="ru")

    def correct(self, text : str):
        encodings = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encodings, forced_bos_token_id=self.tokenizer.get_lang_id("ru"))
        answer = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return str(answer[0]).strip('Secret')



def replace_org(x, leave_org = None):
    doc = Doc(x)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    return_text = x
    for span in doc.spans:
        if span.type == 'ORG':
            if (leave_org is not None) and (span.text in leave_org):
                continue
            else:
                word = 'ORG'
                return_text = return_text.replace(span.text, word)
    return return_text

def prepare_russian_text(text):
    raw_text = re.sub(r'\d+', '0' , text)
    doc = Doc(raw_text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    doc.tag_morph(morph_tagger)
    prepared_text = []
    for token in doc.tokens:
        skip_pos = ['PUNCT', 'ADP', 'SCONJ', 'CCONJ', 'PRON', 'SYM', 'NUM', 'PROPN']
        if token.pos not in skip_pos:
            if token.text in ['ORG']:
                prepared_text.append(token.text)
            else:
                try:
                    token.lemmatize(morph_vocab)
                    prepared_text.append(token.lemma.lower())
                except Exception as ex:
                    prepared_text.append(token.text.lower())
    return prepared_text

