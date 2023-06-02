"""
This mod fine-tunes a BERT model on the ACARIS dataset for comparison with ACARISMdl.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer, AdamW, EarlyStoppingCallback, PreTrainedModel, DistilBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import wandb
import huggingface_hub
import os
import random
import numpy as np

config = {
	"mdl": "distilbert-base-uncased",
	"epochs": 5,
	"batchSize": 14,
	"maxLen": 512,
	"warmupSteps": 0.1, # proportion of total steps, NOT absolute
	"weightDecay": 0.02,
	"outputDir": "./output",
	"earlyStopping": True,
	"earlyStoppingPatience": 2,
	"dropout": 0.1,
	"initlr": 5e-5,
	"epsilon": 1e-8
}

wandb.init(project="simtoon_classifier", entity="simtoonia", config=config)


def lockSeed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True

#0 disabled, as determinism is not guaranteed and lowers performance
#lockSeed(69) # setting a fixed seed for *some* reproducibility

class DistilBertForMulticlassSequenceClassification(DistilBertForSequenceClassification):
	def __init__(self, config):
		super().__init__(config)

	def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		outputs = self.distilbert(input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)

		hidden_state = outputs[0]
		pooled_output = hidden_state[:, 0]
		pooled_output = self.pre_classifier(pooled_output)
		pooled_output = nn.ReLU()(pooled_output)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		loss = None
		if labels is not None:
			lossFct = nn.CrossEntropyLoss()
			loss = lossFct(logits.view(-1, self.num_labels), labels.view(-1))

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



class SIMTOONCLASSIFIER:
	def __init__(self, trainPath, usr):
		self.trainPath = trainPath
		self.usr = usr
		self.tokenizer = DistilBertTokenizerFast.from_pretrained(config["mdl"])
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(config["mdl"], num_labels=2, id2label={0: "neg", 1: "pos"}, label2id={"neg": 0, "pos": 1}, dropout=config["dropout"], attention_dropout=config["dropout"])

	def read_data(self, path):
		df = pd.read_csv(path, sep="|", usecols=["content", "uid"])
		return Dataset.from_pandas(df)
	
	def tokenize_data(self, dataset):
		usrMapping = {self.usr: 1}
		tokenized = dataset.map(
			lambda x: {
				**self.tokenizer(x["content"], truncation=True, padding="max_length", max_length=config["maxLen"]),
				"labels": torch.tensor([usrMapping[usr] if usr in usrMapping else 0 for usr in x["uid"]])
			},
			batched=True,
			remove_columns=["content", "uid"]
		)
		tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
		return tokenized
	
	def get_data_loaders(self, trainDS, valDS):
		trainLoader = DataLoader(trainDS, batch_size=config["batchSize"], shuffle=False)
		valLoader = DataLoader(valDS, batch_size=config["batchSize"], shuffle=False)
		return trainLoader, valLoader
	
	def compute_metrics(self, evalPred):
		logits, labels = evalPred
		preds = torch.argmax(torch.Tensor(logits), dim=1)
		probs = torch.nn.functional.softmax(torch.Tensor(logits), dim=1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
		accuracy = accuracy_score(labels, preds)
		rocAUC = roc_auc_score(labels, probs, multi_class="ovr")
		metrics = {
			"accuracy": accuracy,
			"roc_auc": rocAUC
		}
		metricNames = ["precision", "recall", "f1"]
		labelNames = ["neg", "pos"]
		for metricName, metricValue in zip(metricNames, [precision, recall, f1]):
			for labelName, value in zip(labelNames, metricValue):
				metrics[f"{metricName}_{labelName}"] = float(value)
		return metrics
	
	def train(self):
		trainDS = self.tokenize_data(self.read_data(self.trainPath))
		valDS = self.tokenize_data(self.read_data(self.valPath))

		totalSteps = len(trainDS) // config["batchSize"] * config["epochs"]
		warmupSteps = int(totalSteps * config["warmupSteps"])
		
		trainingArgs = TrainingArguments(
			output_dir=config["outputDir"],
			num_train_epochs=config["epochs"],
			per_device_train_batch_size=config["batchSize"],
			per_device_eval_batch_size=config["batchSize"],
			warmup_steps=warmupSteps,
			weight_decay=config["weightDecay"],
			logging_dir="./logs",
			logging_steps=100,
			learning_rate=config["initlr"],
			evaluation_strategy="epoch",
			save_strategy="epoch",
			load_best_model_at_end=True,
			metric_for_best_model="accuracy",
			save_total_limit=5,
			adam_epsilon=config["epsilon"],
			report_to="wandb",
			fp16=True
		)
		
		trainer = Trainer(
			model=self.model,
			args=trainingArgs,
			train_dataset=trainDS,
			eval_dataset=valDS,
			compute_metrics=self.compute_metrics,
			callbacks=[EarlyStoppingCallback(early_stopping_patience=config["earlyStoppingPatience"])]
		)
		print(f"Number of parameters: {trainer.model.num_parameters()}")
		print("Running eval ...")
		trainer.evaluate()
		print("Running training ...")
		trainer.train()
		print("Saving model ...")
		trainer.save_model(config["outputDir"])
		
		
if __name__ == "__main__":
	classifier = SIMTOONCLASSIFIER("./datasets/train.csv", "simtoon")
	classifier.train()
	wandb.finish()