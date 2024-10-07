import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, TrainerCallback
from sklearn.metrics import accuracy_score, confusion_matrix
from seqeval.metrics import f1_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Step 1: Read the Excel file
data = pd.read_excel('/Users/johnsnow/Desktop/科技政策/NER大样本测试.xlsx')

# Assuming the columns are 'text' and 'label'
texts = data['text'].tolist()
labels = data['label'].tolist()

# Define the label list (example: 'O', 'B-ORG', 'I-ORG', etc.)
label_list = ["O", "政府", "企业", "公民", "组织"]  # Adjust based on your specific labels
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Step 2: Preprocess the data
tokenizer = RobertaTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx].split()  # Assuming labels are space-separated
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")

        # Create label ids
        label_ids = [-100] * self.max_length  # Initialize with -100 (ignored label)
        tokens = self.tokenizer.tokenize(text)
        label_ids[:len(tokens)] = [self.label_to_id[label] for label in labels]

        encoding['labels'] = torch.tensor(label_ids, dtype=torch.long)

        return {k: v.squeeze() if v.dim() > 1 else v for k, v in encoding.items()}

# Prepare dataset
dataset = NERDataset(texts, labels, tokenizer, label_to_id)

# Step 3: Prepare the data loader
data_collator = DataCollatorForTokenClassification(tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=data_collator)

# Step 4: Fine-tune the RoBERTa model
model = RobertaForTokenClassification.from_pretrained('hfl/chinese-roberta-wwm-ext', num_labels=len(label_list))

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=50,
    weight_decay=0.005,
    logging_dir='./logs',
    report_to='none'  # to avoid reporting to third-party platforms
)

# Function to compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = [[id_to_label[id_] for id_ in label if id_ != -100] for label in labels]
    true_predictions = [[id_to_label[pred] for pred, label in zip(prediction, label_ids) if label != -100] 
                        for prediction, label_ids in zip(predictions, labels)]
    
    accuracy = accuracy_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    report = classification_report(true_labels, true_predictions)
    
    # Flatten the lists for confusion matrix calculation
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    
    conf_matrix = confusion_matrix(flat_true_labels, flat_true_predictions, labels=label_list)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()  # Convert ndarray to list
    }

# Custom callback to log training loss
class LogTrainingLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"Training loss: {logs['loss']}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LogTrainingLossCallback]
)

trainer.train()

# Step 5: Evaluate the model
metrics = trainer.evaluate()

# Print the evaluation metrics
print("Training loss:", metrics['eval_loss'])
print("Accuracy:", metrics['eval_accuracy'])
print("F1 Score:", metrics['eval_f1'])
print("Classification Report:\n", metrics['eval_classification_report'])
print("Confusion Matrix:\n", metrics['eval_confusion_matrix'])
