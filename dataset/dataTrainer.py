from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import get_peft_model, LoraConfig, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "taronklm/Qwen2.5-0.5B-Instruct-lora-chatbot"

logger.info("loading dataset...")
# ds = load_dataset("json", data_files=r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\chatbot-backend\dataset\dataset_v3_transformed.json")

ds = load_dataset("taronklm/optimization_and_creation")
print(ds["train"][0])

logger.info("loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(example):
    user_message = next((msg["content"] for msg in example["messages"] if msg["role"] == "user"), "")
    assistant_message = next((msg["content"] for msg in example["messages"] if msg["role"] == "assistant"), "")

    # Tokenize input and output
    model_inputs = tokenizer(user_message, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(assistant_message, max_length=512, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]  # Add labels for training
    return model_inputs

logger.info("tokenize dataset...")
tokenized_dataset = ds.map(tokenize_function, batched=False)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

logger.info("configure peft (lora)...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)


model = get_peft_model(model, peft_config)
# model = PeftModel.from_pretrained(model, r"C:\Users\Taro\Desktop\bachelorarbeit\code\bot\Qwen\Qwen2.5-0.5B-Instruct-lora-chatbot\checkpoint-12")

model.print_trainable_parameters()

logger.info("define training args...")
training_args = TrainingArguments(
    output_dir= model_name + "-lora-chatbot",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=4,
    save_steps=10, 
    learning_rate=1e-4,
    save_on_each_node=True,
    # resume_from_checkpoint=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

logger.info("start training...")
trainer.train(resume_from_checkpoint=None)

from huggingface_hub import login

logger.info("uploading model...")
write_key = "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"
login(write_key)

hf_name ="taronklm"
model_id = hf_name + "/Prompto" 
model.push_to_hub(model_id)
trainer.push_to_hub(model_id)