import torch
import torch.nn.functional as F
from dataset import prepare_text


KAD_CAT_INFO_PATH = 'kad_categs_info.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_last_hidden_state_embedding(encoding, model):
    with torch.no_grad():
        last_hidden_state = model.bert(
            input_ids=encoding['input_ids'], 
            attention_mask=encoding['attention_mask']
        )['last_hidden_state']
    emb = last_hidden_state.mean(dim=1).squeeze(0)
    return emb

def predict_single(clf_model, encoding):
    clf_model = clf_model.eval()
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = clf_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    proba = F.softmax(outputs, dim=1)
    preds = torch.argmax(proba, dim=1).item()
    return preds, proba
    
def single_pipeline(clf_model, tokenizer, raw_text):
    cleaned_text, encoding = prepare_text(raw_text, tokenizer)
    bert_id_pred, bert_proba = predict_single(clf_model, encoding)

    return bert_id_pred, bert_proba, encoding, cleaned_text
