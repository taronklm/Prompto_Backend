from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer
)
import logging
import json
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model name
model_name = "distilgpt2"

logger.info("Loading dataset...")
df = pd.read_json(r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\chatbot-backend\dataset\dataset_v3_transformed.json")

print("DATAFRAME: ", df)

df["text"] = df["messages"]

train_df, test_df = train_test_split(df, test_size=0.1)

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

logger.info("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Combine the role and content into a single string for each message
    texts = []
    for example in examples["text"]:  # Iterate over each batch of examples (list of messages)
        # Example is a list of messages (dictionaries)
        conversation = " ".join([f"{msg['role']}: {msg['content']}" for msg in example])
        texts.append(conversation)
    
    # Tokenize the concatenated text
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

logger.info("Define training arguments...")
training_args = TrainingArguments(
    output_dir=f"{model_name}-chatbot",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=10_000,
    learning_rate=5e-5,
    save_total_limit=2,
    logging_dir="./logs",
    eval_strategy="epoch",
    eval_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

logger.info("Start training...")
trainer.train()

from huggingface_hub import login

logger.info("Uploading model...")
write_key = "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"
login(write_key)

hf_name = "taronklm"
model_id = f"{hf_name}/distilgpt2-chatbot"
# model.push_to_hub(model_id)
tokenizer.push_to_hub(model_id)
trainer.push_to_hub(model_id)

print(f"Tokenzer vocab size: {len(tokenizer)}")
print(f"Model embedding size: {model.transformer.wte.weight.size(0)}")