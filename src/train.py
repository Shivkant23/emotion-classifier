import wandb

from transformers import (
    Trainer,
    TrainingArguments
)

from dataset import (
    load_emotion_dataset,
    tokenize_dataset
)

from model import create_model

from metrics import compute_metrics


def train_model(
    run_name,
    learning_rate,
    batch_size,
    epochs,
    weight_decay
):

    wandb.init(
        project="emotion-classification",
        name=run_name,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay
        }
    )

    dataset = load_emotion_dataset()

    tokenized_dataset = tokenize_dataset(dataset)

    training_args = TrainingArguments(
        output_dir=f"./outputs/{run_name}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to="wandb",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=create_model(),
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    metrics = trainer.evaluate()

    print(metrics)

    wandb.finish()

    