from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

peft_model_id = "outputs/final_model"
peft_config = PeftConfig.from_pretrained(peft_model_id)

base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

dataset = load_dataset("imdb")["test"].select(range(1000))

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512 - peft_config.num_virtual_tokens).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

labels = []
preds = []

for row in dataset:
    labels.append(row["label"])
    preds.append(predict(row["text"]))

print(f"Accuracy: {accuracy_score(labels, preds):.4f}")
print(f"F1 Score: {f1_score(labels, preds):.4f}")
