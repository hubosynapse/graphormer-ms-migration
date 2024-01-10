import os
import json
from os.path import join as pjoin

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, ops, context

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy
from mindnlp import load_dataset
from mindnlp.transformers import (
    GraphormerForGraphClassification,
    GraphormerDataCollator
)


def jsonl_file_iterator(file_path, column_names):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            jline = json.loads(line)
            values = []
            for cn in column_names:
                if cn not in jline:
                    ValueError(f"Key {cn} not contained in {line} in the input json file")
                values.append(jline[cn])

            yield values


class GraphDataset(GeneratorDataset):

    def __init__(self, json_path, column_names, **kwargs):
        generator = jsonl_file_iterator(json_path, column_names)
        super().__init__(generator, column_names, **kwargs)



if __name__ == "__main__":
    # data_dir = "../../graphormer-data/ogbg-molhiv/"
    # column_names = ["edge_index", "edge_attr", "y", "num_nodes", "node_feat"]
    # dataset_test = GraphDataset(pjoin(data_dir, 'test.jsonl'), column_names)

    dataset = load_dataset("../../graphormer-data/ogbg-molhiv/")
    dataset_train = dataset["train"]
    dataset_val = dataset["validation"]
    column_names = ["edge_index", "edge_attr", "y", "num_nodes", "node_feat"]

    data_collator = GraphormerDataCollator(on_the_fly_processing=True)

    dataset_train = dataset_train.batch(batch_size=3,
                                        per_batch_map=data_collator,
                                        input_columns=column_names,
                                        output_columns=data_collator.output_columns)

    dataset_val = dataset_val.batch(batch_size=3,
                                    per_batch_map=data_collator,
                                    input_columns=column_names,
                                    output_columns=data_collator.output_columns)

    metric = Accuracy()
    ckpoint_cb = CheckpointCallback(save_path='checkpoint',
                                    ckpt_name='graphormer',
                                    epochs=1,
                                    keep_checkpoint_max=2)

    best_model_cb = BestModelCallback(save_path='checkpoint',
                                      ckpt_name='graphormer',
                                      auto_load=True)

    model_dir = "../../pretrained_models/graphormer/graphormer-base-pcqm4mv2/"
    # model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2", from_pt=True)
    # model.save_pretrained("../../pretrained_models/graphormer/graphormer-base-pcqm4mv2/")
    model = GraphormerForGraphClassification.from_pretrained(model_dir)

    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=5e-5,
                                   beta1=0.9,
                                   beta2=0.999,
                                   eps=1e-8)

    trainer = Trainer(network=model,
                      train_dataset=dataset_train,
                      eval_dataset=dataset_val,
                      metrics=metric,
                      epochs=1,
                      optimizer=optimizer,
                      callbacks=[ckpoint_cb, best_model_cb],
                      jit=False)

    # trainer.set_amp(level='O1') # Mixed-precision Training



    # start training
    trainer.run(tgt_columns="labels")
