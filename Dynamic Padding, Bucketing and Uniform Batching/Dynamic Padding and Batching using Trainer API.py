import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Optional
from typing import List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, EvalPrediction, Trainer, HfArgumentParser, TrainingArguments, \
    AutoModelForSequenceClassification, set_seed, AutoConfig
from transformers import PreTrainedTokenizer, DataCollator, PreTrainedModel
import wandb

set_seed(123)

label_codes = {"contradictory": 0, "contradiction": 0, "neutral": 1, "entailment": 2}

wandb.init(project="speed_training")


class MyTrainer(Trainer):
    def _setup_wandb(self):
        wandb.init(project="speed_training",
                   config=vars(self.args),
                   name=self.args.output_dir)
        wandb.watch(self.model, log="gradients", log_freq=self.args.logging_steps)


@dataclass
class Example:
    text_a: str
    text_b: str
    label: int


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    label: int


@dataclass
class ModelParameters:
    max_seq_len: Optional[int] = field(
        default=None,
        metadata={"help": "max seq len"},
    )
    dynamic_padding: bool = field(
        default=False,
        metadata={"help": "limit pad size at batch level"},
    )
    smart_batching: bool = field(
        default=False,
        metadata={"help": "build batch of similar sizes"},
    )
    dynamic_batch_size: bool = field(
        default=False,
        metadata={"help": "build batch of similar sizes"},
    )


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_max_length: bool, max_len: int,
                 examples: List[Example]) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples: List[Example] = examples
        self.current = 0
        self.pad_to_max_length = pad_to_max_length

    def encode(self, ex: Example) -> Features:
        encode_dict = self.tokenizer.encode_plus(text=ex.text_a,
                                                 text_pair=ex.text_b,
                                                 add_special_tokens=True,
                                                 max_length=self.max_len,
                                                 pad_to_max_length=self.pad_to_max_length,
                                                 return_token_type_ids=False,
                                                 return_attention_mask=True,
                                                 return_overflowing_tokens=False,
                                                 return_special_tokens_mask=False,
                                                 )
        return Features(input_ids=encode_dict["input_ids"],
                        attention_mask=encode_dict["attention_mask"],
                        label=ex.label)

    def __getitem__(self, idx) -> Features:  # Trainer doesn't support IterableDataset (define sampler)
        if self.current == len(self.examples):
            self.current = 0
        ex = self.examples[self.current]
        self.current += 1
        return self.encode(ex=ex)

    def __len__(self):
        return len(self.examples)


def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
    return seq + (max_batch_len - len(seq)) * [pad_value]


@dataclass
class SmartCollator(DataCollator):
    pad_token_id: int

    def collate_batch(self, batch: List[Features]) -> Dict[str, torch.Tensor]:
        batch_inputs = list()
        batch_attention_masks = list()
        labels = list()
        max_size = max([len(ex.input_ids) for ex in batch])
        for item in batch:
            batch_inputs += [pad_seq(item.input_ids, max_size, self.pad_token_id)]
            batch_attention_masks += [pad_seq(item.attention_mask, max_size, 0)]
            labels.append(item.label)

        return {"input_ids": torch.tensor(batch_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_masks, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
                }

def load_transformers_model(pretrained_model_name_or_path: str,
                            use_cuda: bool,
                            mixed_precision: bool) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                        num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        config=config)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)

    if mixed_precision:
        try:
            from apex import amp
            model = amp.initialize(model, opt_level='O1')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return model


def load_train_data(path: str, sort: bool) -> List[Example]:
    sentences = list()
    with open(path) as f:
        first = False
        for line in f:
            if not first:
                first = True
                continue
            text_a, text_b, label = line.rstrip().split("\t")
            lab = len(text_a) + len(text_b)
            sentences.append((lab, Example(text_a=text_a, text_b=text_b, label=label_codes[label])))
    if sort:
        sentences.sort(key=lambda x: x[0])

    return [e for (_, e) in sentences]


def load_dev_data(path: str) -> List[Example]:
    sentences = list()
    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("\t")
            if line[0] != "fr":
                continue
            text_a = line[6]
            text_b = line[7]
            label = line[1]
            lab = len(text_a) + len(text_b)
            sentences.append((lab, Example(text_a=text_a, text_b=text_b, label=label_codes[label])))

    sentences.sort(key=lambda x: x[0])

    return [e for (_, e) in sentences]


def build_batches(sentences: List[Example], batch_size: int) -> List[Example]:
    batch_ordered_sentences = list()
    while len(sentences) > 0:
        to_take = min(batch_size, len(sentences))
        select = random.randint(0, len(sentences) - to_take)
        batch_ordered_sentences += sentences[select:select + to_take]
        del sentences[select:select + to_take]
    return batch_ordered_sentences


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelParameters))
    training_args, model_args = parser.parse_args_into_dataclasses()  # type: (TrainingArguments, ModelParameters)

    train_sentences = load_train_data(path="resources/XNLI-MT-1.0/multinli/multinli.train.fr.tsv",
                                      sort=model_args.smart_batching)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="camembert-base")
    if model_args.max_seq_len:
        max_sequence_len = model_args.max_seq_len
    else:
        longest_sentence = max(train_sentences, key=len)
        max_sequence_len = len(tokenizer.encode(text=longest_sentence.text_a, text_pair=longest_sentence.text_b))

    train_batches = build_batches(sentences=train_sentences, batch_size=training_args.per_gpu_train_batch_size)
    valid_sentences = load_dev_data(path="resources/XNLI-1.0/xnli.test.tsv")
    valid_batches = build_batches(sentences=valid_sentences, batch_size=training_args.per_gpu_eval_batch_size)

    train_set = TextDataset(tokenizer=tokenizer,
                            max_len=max_sequence_len,
                            examples=train_batches,
                            pad_to_max_length=not model_args.dynamic_padding)

    valid_set = TextDataset(tokenizer=tokenizer,
                            max_len=max_sequence_len,
                            examples=valid_batches,
                            pad_to_max_length=not model_args.dynamic_padding)

    model = load_transformers_model(pretrained_model_name_or_path="camembert-base",
                                    use_cuda=True,
                                    mixed_precision=False)


    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}


    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        # data_collator=IdentityCollator(pad_token_id=tokenizer.pad_token_id),
        data_collator=SmartCollator(pad_token_id=tokenizer.pad_token_id),
        tb_writer=SummaryWriter(log_dir='logs', flush_secs=10),
        eval_dataset=valid_set,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    wandb.config.update(model_args)
    wandb.config.update(training_args)
    wandb.log({"training time": int((time.time() - start_time) / 60)})
    trainer.save_model()
    trainer.evaluate()
    logging.info("*** Evaluate ***")
    result = trainer.evaluate()
    wandb.log(result)

    output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logging.info("***** Eval results *****")
        for key, value in result.items():
            logging.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))
