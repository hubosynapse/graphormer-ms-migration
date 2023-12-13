import numpy as np
import evaluate

from datasets import load_dataset
from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator
from transformers import TrainingArguments, Trainer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from peft import LoraConfig, get_peft_model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


if __name__ == "__main__":

    model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        )

    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)


    dataset = load_dataset("ogb/ogbg-molhiv")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    data_collator = GraphormerDataCollator(on_the_fly_processing=True)


    training_args = TrainingArguments(output_dir="test_trainer",
                                      remove_unused_columns=False,
                                      per_device_train_batch_size=4,
                                      fp16=True,
                                      evaluation_strategy="epoch")

    metric = evaluate.load("accuracy")

    trainer = Trainer(model=lora_model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics,
                      data_collator=data_collator)

    trainer.train()
