import os
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from datasets import Dataset
import matplotlib.pyplot as plt

# Disable W&B
os.environ["WANDB_DISABLED"] = "true"

# Load data
df_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    return text

df_train["text"] = df_train["text"].apply(preprocess_text)
df_test["text"] = df_test["text"].apply(preprocess_text)


X_train, X_val, y_train, y_val = train_test_split(df_train["text"], df_train["target"], test_size=0.2, random_state=42)

# Convert to datasets
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
val_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_val, 'label': y_val}))

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=80)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch", 
    logging_dir="./logs",
    logging_steps=10,
)

# Define metrics
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    acc = accuracy_score(labels, pred)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()


trainer.evaluate()

# Function to display confusion matrix
def displayConfusionMatrix(y_true, y_pred, dataset):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Not Disaster","Disaster"],
        cmap=plt.cm.Blues
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1_score = tp / (tp + ((fn + fp) / 2))
    disp.ax_.set_title("Confusion Matrix on " + dataset + " Dataset -- F1 Score: " + str(f1_score.round(2)))
    plt.show()

# Predict and display confusion matrix for training data
train_preds = trainer.predict(train_dataset)
y_pred_train = np.argmax(train_preds.predictions, axis=1)
displayConfusionMatrix(y_train, y_pred_train, "Training")

# Predict and display confusion matrix for validation data
val_preds = trainer.predict(val_dataset)
y_pred_val = np.argmax(val_preds.predictions, axis=1)
displayConfusionMatrix(y_val, y_pred_val, "Validation")
