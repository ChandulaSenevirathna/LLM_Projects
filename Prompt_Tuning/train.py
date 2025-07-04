from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model
from data.Load_dataset import get_dataset
from models.prompt_tuning_config import get_prompt_config
from sklearn.metrics import accuracy_score, f1_score
import torch

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
    }

def main():
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    peft_config = get_prompt_config()
    model = get_peft_model(base_model, peft_config)

    num_virtual_tokens = peft_config.num_virtual_tokens
    effective_max_length = 512 - num_virtual_tokens

    dataset = get_dataset(tokenizer, max_length=effective_max_length)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    training_args = TrainingArguments(
        output_dir="./outputs",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir="./logs",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.model.save_pretrained("outputs/final_model")

if __name__ == "__main__":
    main()
