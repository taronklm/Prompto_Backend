from datasets import load_dataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("LOADING DATASET...")
ds = load_dataset("taronklm/Prompto_v6", split="train")

logger.info("SPLITTING DATASET...")
split_ratio = 0.8
train_ds, eval_ds = ds.train_test_split(test_size=1 - split_ratio, seed=42).values()

ds = ds.map(remove_columns="split")
print(ds[0])
logger.info("SHUFFLE DATASET...")
ds = ds.shuffle()
print(ds[0])

from transformers import AutoTokenizer

logger.info("SETTING MODEL NAME AND LOADING TOKENIZER...")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

from transformers import AutoModelForCausalLM
import torch

def apply_chat_template(example,tokenizer):
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return example


logger.info("LOADING MODEL...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32,)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

logger.info("PREPROCESSING TRAINING DATASET...")
processed_train_dataset = train_ds.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,
    remove_columns=list(train_ds.features),
    desc="Applying chat template to train_sft",
)

logger.info("PREPROCESSING EVALUATION DATASET...")
processed_eval_dataset = eval_ds.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=1,
    remove_columns=list(eval_ds.features),
    desc="Applying chat template to eval_sft",
)

logger.info("PROCESSED DATA 0...")
print(processed_train_dataset[0])

logger.info("DATASET LENGTH...")
print(len(processed_train_dataset))

logger.info("DEFINE LORA CONFIG...")
logging_steps = max(1, len(processed_train_dataset) // 10)
logger.info("DEFINE SFTCONFIG...")

print(f"Training dataset size: {len(processed_train_dataset)}")
print(f"Evaluation dataset size: {len(processed_eval_dataset)}")

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    packing=True,
    max_seq_length=128,
    output_dir="./trained_model",
    overwrite_output_dir=True,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=logging_steps,
    dataset_text_field="text",
    use_cpu=True,
    gradient_checkpointing=True,
    save_strategy="steps",
    eval_strategy="epoch",
    eval_steps=10
)

logger.info("DEFINE METRIC FUNCTION")

import numpy as np
from evaluate import load

bertscore = load("bertscore")

def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bertscore_metrics = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="de")

    return {
        "bertscore_precision": np.mean(bertscore_metrics["precision"]),
        "bertscore_recall": np.mean(bertscore_metrics["recall"]),
        "bertscore_f1": np.mean(bertscore_metrics["f1"]),
    }

logger.info("DEFINE SFTTRAINER...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_eval_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    compute_metrics=compute_metrics
)

print(trainer)

logger.info("START TRAINING...")
trainer.train()

logger.info("SAVING MODEL...")
trainer.save_model("./fine_tuned_model_v1")
tokenizer.save_pretrained("./fine_tuned_model_v1")

from huggingface_hub import login

write_key = os.getenv("WRITE_HUGGINGFACE_TOKEN")
login(write_key)

logger.info("UPLOADING MODEL...")
hf_name ="taronklm"
model_id = hf_name + "/Qwen2.5-0.5B-Instruct-chatbot" 
# model.push_to_hub(model_id)
trainer.push_to_hub(model_id)