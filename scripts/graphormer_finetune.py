import os
import json
from os.path import join as pjoin

import mindspore
from mindspore.dataset import text, GeneratorDataset, transforms
from mindspore import nn, context

from mindnlp.engine import Trainer, Evaluator
from mindnlp.engine.callbacks import CheckpointCallback, BestModelCallback
from mindnlp.metrics import Accuracy

# from mindnlp.transformers import (GraphormerForGraphClassification, GraphormerDataCollator)


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
    data_dir = "../../graphormer-data/ogbg-molhiv/"
    column_names = ["edge_index", "edge_attr", "y", "num_nodes", "node_feat"]
    dataset_test = GraphDataset(pjoin(data_dir, 'test.jsonl'), column_names)



    """
    model = GraphormerForGraphClassification.from_pretrained("clefourrier/graphormer-base-pcqm4mv2", from_pt=True)
    model.enable_recompute()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=2e-5)

    metric = Accuracy()
    # define callbacks to save checkpoints
    ckpoint_cb = CheckpointCallback(save_path='checkpoint',
                                    ckpt_name='graphormer',
                                    epochs=1,
                                    keep_checkpoint_max=2)

    best_model_cb = BestModelCallback(save_path='checkpoint',
                                      ckpt_name='graphormer',
                                      auto_load=True)

    trainer = Trainer(network=model,
                      loss_fn=loss_fn,
                      train_dataset=dataset_train,
                      eval_dataset=dataset_val,
                      metrics=metric,
                      epochs=1,
                      optimizer=optimizer,
                      callbacks=[ckpoint_cb, best_model_cb],
                      jit=True)

    trainer.set_amp(level='O1')



    # start training
    trainer.run(tgt_columns="labels")
"""



# def predict(text, label=None):
#     label_map = {0: "消极", 1: "中性", 2: "积极"}

#     text_tokenized = Tensor([tokenizer(text).input_ids])
#     logits = model(text_tokenized)
#     predict_label = logits[0].asnumpy().argmax()
#     info = f"inputs: '{text}', predict: '{label_map[predict_label]}'"
#     if label is not None:
#         info += f" , label: '{label_map[label]}'"
#     print(info)




# from mindspore import Tensor

# for label, text in dataset_infer:
#     predict(text, label)
