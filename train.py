import os
import os.path as osp
import docx
from datasets import Dataset
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments

if __name__ == '__main__':

    # Load tokenizer and model
    checkpoint = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Load Dataset
    data_dir = "letters"
    filepaths = sorted([item.path for item in os.scandir(data_dir)])

    texts = []
    for filepath in filepaths:

        # Open document
        document = docx.Document(filepath)
        # Extract text from document
        text = []
        for paragraph in document.paragraphs:
            if paragraph.text.strip() == "": continue
            text.append(paragraph.text)
        text = "\n\n".join(text)
        # Append EOS to document
        text += tokenizer.eos_token

        # Add text to samples
        texts.append(text)

    dataset = Dataset.from_dict({ "text": texts })
    print(dataset)

    # Preprocess dataset
    def tokenize_function(example):
        return tokenizer(
            example["text"],
            truncation=True,
            add_special_tokens=False,
            max_length=128,
            stride=80,
            return_overflowing_tokens=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(tokenized_dataset)

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # Apply LORA
    lora_config = LoraConfig(r=4, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Define training arguments
    output_dir = "letter"
    training_args = TrainingArguments(
        output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=200,
        learning_rate=1e-3,
        weight_decay=0.01,
        warmup_steps=20,
        save_strategy="epoch")

    # Train model
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
