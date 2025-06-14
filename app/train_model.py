# Complete training script
import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, classification_report

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

# Load dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"].shuffle(seed=42).select(range(2500))
test_dataset = dataset["test"].shuffle(seed=42).select(range(2500))

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Convert to TF datasets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
tf_train_dataset = model.prepare_tf_dataset(tokenized_train, shuffle=True, batch_size=16, collate_fn=data_collator)
tf_test_dataset = model.prepare_tf_dataset(tokenized_test, shuffle=False, batch_size=16, collate_fn=data_collator)

# Compile and train
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

print("Starting training...")
model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=2)

# Save model
model.save("sentiment_model")
tokenizer.save_pretrained("sentiment_model")
print("Model saved to sentiment_model/")