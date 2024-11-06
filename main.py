from model_handler import ModelHandler
from data_handler import DataHandler
from embedding_handler import EmbeddingCreator
from dataset_handler import CustomConversationalDataset, CustomDataCollatorWithPadding
from trainer import CustomTrainer
from utils import ClearCUDACacheCallback
import torch
from transformers import TrainingArguments, EarlyStoppingCallback, GenerationConfig, TrainingArguments
import torch
import time
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

def main():
    handler = ModelHandler()
    original_model = handler.load_seq2seq_model()
    encoder_model = handler.load_encoder_model()
    tokenizer = handler.load_tokenizer()

    conversations = DataHandler.load_data('therapy.json')

    creator = EmbeddingCreator(encoder_model, tokenizer, handler.device)
    embeddings, targets = creator.create_incremental_embeddings(conversations)

    dataset = CustomConversationalDataset(embeddings, targets, tokenizer)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    output_dir = f'./peft-training-{str(int(time.time()))}'

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=32,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
    )

    model_with_lora = get_peft_model(original_model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",  
        learning_rate=5e-3,
        per_device_train_batch_size=4,
        num_train_epochs=50,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=200,
        load_best_model_at_end=True,
        auto_find_batch_size=True,
)

    trainer = CustomTrainer(
        model=original_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), ClearCUDACacheCallback()],
    )

    trainer = CustomTrainer(
        model=model_with_lora,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), ClearCUDACacheCallback()],
        data_collator=CustomDataCollatorWithPadding(),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.save_pretrained('./peft-checkpoint-local')
    tokenizer.save_pretrained('./peft-checkpoint-local')

    prompt = f"""
    Analyze conversation between User and Therapist. Analyize emotional change of User and Therapist. You are Chatbot. Respond like therapist in empathetic way and ask questions when appropriate:

    User: Public speaking makes me so nervous; I avoid it whenever I can. (Emotion: Anxiety, Fear)

    Chatbot: 
    """
    print(prompt)

    # Ensure the model is on the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    peft_model = PeftModel.from_pretrained(original_model, 
                                        "./peft-dialogue-summary-checkpoint-local", 
                                        torch_dtype=torch.bfloat16,
                                        is_trainable=False)
    peft_model = peft_model.to(device)


    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Define the generation configuration with adjusted parameters
    gen_config = GenerationConfig(
        max_new_tokens=512,
            min_length=20,
            num_beams=10,
            temperature=0.2,
            repetition_penalty=1.1,
            early_stopping=False,
            length_penalty=1.5
    )

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=gen_config)
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    print(peft_model_text_output)

if __name__ == "__main__":
    main()
