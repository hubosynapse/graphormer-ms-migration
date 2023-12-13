import numpy as np
import evaluate

from datasets import load_dataset
from transformers import GraphormerForGraphClassification
from transformers.models.graphormer.collating_graphormer import GraphormerDataCollator
from transformers import TrainingArguments, Trainer

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":

    dataset = load_dataset("ogb/ogbg-molhiv")

    train_dataset = dataset["train"]

    data_collator = GraphormerDataCollator(on_the_fly_processing=True)

    model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2")

    training_args = TrainingArguments(output_dir="test_trainer",
                                      remove_unused_columns=False,
                                      per_device_train_batch_size=4,
                                      fp16=True,
                                      evaluation_strategy="epoch")

    metric = evaluate.load("accuracy")

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      data_collator=data_collator)

    trainer.train()
