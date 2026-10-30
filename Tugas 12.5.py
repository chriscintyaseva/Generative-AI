
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel
)
from datasets import Dataset
import torch

# 1. Load base model
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none"
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# 3. Prepare dataset
training_examples = [
    "Question: What is RAG? Answer: RAG stands for Retrieval-Augmented Generation.",
    "Question: What is fine-tuning? Answer: Fine-tuning trains a pre-trained model on new data.",
    "Question: What is an embedding? Answer: An embedding is a dense vector representing text meaning.",
    "Question: What is a vector database? Answer: A vector database stores embeddings for similarity search.",
    "Question: What is LangChain? Answer: LangChain is a framework for building LLM applications."
]

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

dataset = Dataset.from_dict({"text": training_examples})

tokenized = dataset.map(tokenize, batched=True)

# Prepare labels and ignore padding tokens
def prepare_labels(example):
    example["labels"] = [
        token if mask == 1 else -100
        for token, mask in zip(
            example["input_ids"],
            example["attention_mask"]
        )
    ]
    return example

tokenized = tokenized.map(prepare_labels)

# 4. Training arguments
output_path = "E:/KULIAH!!!!!!/GEN AI/Modul 12/lora_output"
adapter_path = "E:/KULIAH!!!!!!/GEN AI/Modul 12/lora_adapter"

training_args = TrainingArguments(
    output_dir=output_path,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    use_cpu=True
)

# 5. Train the model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized
)

trainer.train()

# 6. Save LoRA adapter
peft_model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)

print("\nLoRA adapter saved successfully.")

# 7. Load adapter for inference
base = AutoModelForCausalLM.from_pretrained(model_name)

merged = PeftModel.from_pretrained(
    base,
    adapter_path
)

merged.eval()

# Prepare input
inputs = tokenizer(
    "Question: What is RAG? Answer:",
    return_tensors="pt"
)

# Generate answer
with torch.no_grad():
    out = merged.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=30,
        pad_token_id=tokenizer.eos_token_id
    )

print("\nGenerated Answer:")
print(tokenizer.decode(out[0], skip_special_tokens=True))
