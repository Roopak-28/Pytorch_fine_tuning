import argparse
import os
import shutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import get_peft_model, LoraConfig, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_uri", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--training_logs", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.trained_model, exist_ok=True)
    os.makedirs(args.training_logs, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.dataset_name, split="train[:3000]")
    dataset = dataset.rename_column("label", "labels")

    # Tokenize and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_uri)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_uri, num_labels=5)

    # Apply LoRA
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, config)

    def preprocess(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(preprocess, batched=True)

    args_train = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=dataset
    )
    trainer.train()

    # Save model output
    for f in os.listdir(args.output_dir):
        src = os.path.join(args.output_dir, f)
        dst = os.path.join(args.trained_model, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    with open(os.path.join(args.training_logs, "log.txt"), "w") as f:
        f.write("Training completed successfully.")

if __name__ == "__main__":
    main()
