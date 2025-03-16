import os
import random
import re
import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# Removed extra TensorBoard SummaryWriter since Trainer logs to TensorBoard automatically

#############################################################################
#                           Summarization Functions
#############################################################################

def clean_text(text):
    """Remove unwanted special tokens from text."""
    return text.replace("<extra_id_0>", "").strip()

def post_process_generated_text(text):
    """Post-process generated text to remove unwanted tokens and extra whitespace."""
    text = text.replace("<extra_id_0>", "").strip()
    return " ".join(text.split())

def preprocess_function(examples, tokenizer, body_key, summary_key, max_input_len=512, max_target_len=256, chunk_overlap=50):
    """Prepares dataset: Tokenizes input with chunking and cleans target summaries."""
    chunked_inputs = []
    chunked_summaries = []

    for body_text, summary_text in zip(examples[body_key], examples[summary_key]):        # Skip if summary is too short
        if len(summary_text.split()) < 50:
            continue

        # Clean the summary text to remove unwanted tokens
        summary_text = clean_text(summary_text)
        tokenized_body = tokenizer(body_text, truncation=False)["input_ids"]
        body_chunks = [
            tokenized_body[i : i + max_input_len]
            for i in range(0, len(tokenized_body), max_input_len - chunk_overlap)
        ]
        tokenized_summary = tokenizer(summary_text, truncation=True, max_length=max_target_len)["input_ids"]

        for chunk in body_chunks:
            chunked_inputs.append(chunk)
            chunked_summaries.append(tokenized_summary)

    return {"input_ids": chunked_inputs, "labels": chunked_summaries}

def get_rouge_scores(model, dataset, tokenizer, device, body_key="body", summary_key="summary", max_length=128, num_beams=3):
    """Evaluate a model by generating summaries and comparing with reference summaries.
       Uses bad_words_ids to prevent generation of <extra_id_0>."""
    debug = True

    rouge = evaluate.load("rouge")
    preds, refs = [], []
    
    # Get the token id for <extra_id_0>
    bad_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    bad_words = [[bad_token_id]]

    for i, ex in enumerate(dataset):
        body_text = ex[body_key]
        ref_text  = ex[summary_key]
        if not body_text.strip():
            preds.append("")
            refs.append(ref_text)
            continue

        input_ids = tokenizer.encode("summarize: " + body_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Avoid repeated words
            do_sample=False,  # Use deterministic generation
            temperature=0.7,  # Ensure stable output
            top_k=50,  # Prevent degenerate outputs
            top_p=0.95, # Ensure diverse summaries
            bad_words_ids=bad_words 
        )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_text = post_process_generated_text(pred_text)
        pred_text = post_process_generated_text(pred_text)
        preds.append(pred_text)
        refs.append(ref_text)

        # Print a few debug examples
        if debug and i < 3:
            print(f"\n--- Debug Example {i} ---")
            print("Input (first 300 chars):", body_text[:300])
            print("Predicted Summary:", pred_text)
            print("Reference Summary:", ref_text)
            print("-" * 50)

    result = rouge.compute(predictions=preds, references=refs)
    # Convert floats to percentages if needed
    if isinstance(result["rouge1"], float):
        return {k: v * 100 for k, v in result.items()}
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom Trainer to handle LoRA-specific issues."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

def train_lora(base_model, dataset, tokenizer, model_repo_id, 
               body_key="body", summary_key="summary", 
               num_epochs=4, learning_rate=1e-4, skip_if_hf_exists=True,
               freeze_base=False):
    """Fine-tunes a model using LoRA, checks HF repo to skip training if already exists,
       and optionally freezes the base model parameters (non-adapter) before training."""
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Check if model exists on Hugging Face
    api = HfApi()
    try:
        info = api.repo_info(model_repo_id, repo_type="model")
        print(f"\n[Skipping Training?] {model_repo_id} found on HF. Checking for adapter config...")
        
        # This will try to load the adapter config and weights.
        loaded_lora_model = PeftModel.from_pretrained(base_model, model_repo_id)
        print("\n=== LoRA Model Successfully Loaded ===")
        # print(loaded_lora_model)  # Debug info
        print(f"Found LoRA adapter in {model_repo_id}, skipping training.")
        
        loaded_lora_model.to(device)
        return loaded_lora_model
    except (RepositoryNotFoundError, ValueError, OSError) as e:
        print(f"HF repo {model_repo_id} found but no valid LoRA adapter inside (or missing adapter_config.json).")
        print(f"Proceeding with training. Error was: {e}")

    base_model.to(device)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=1, lora_alpha=16, lora_dropout=0.2,
        target_modules=["q", "v"]
    )
    lora_model = get_peft_model(base_model, peft_config).to(device)

    # If freeze_base is True, freeze all parameters except those related to LoRA
    if freeze_base:
        for name, param in lora_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        print("Base model parameters frozen. Only LoRA adapter parameters will be updated.")

    # Handle dataset splits: if dataset is not a dict, create splits
    if isinstance(dataset, dict):
        train_ds = dataset["train"]
        eval_ds = dataset["validation"]
        test_ds = dataset["test"]
    else:
        splits = dataset.train_test_split(test_size=0.2)
        eval_test = splits["test"].train_test_split(test_size=0.5)
        train_ds = splits["train"]
        eval_ds = eval_test["train"]
        test_ds = eval_test["test"]

    # Tokenize each split separately using their own column names
    def tokenize_dataset(ds):
        return ds.map(lambda x: preprocess_function(x, tokenizer, body_key, summary_key),
                      batched=True,
                      remove_columns=ds.column_names)
    
    tokenized_train = tokenize_dataset(train_ds)
    tokenized_eval = tokenize_dataset(eval_ds)
    tokenized_test = tokenize_dataset(test_ds)
    
    # Prepare a dictionary for consistency
    tokenized_ds = {"train": tokenized_train, "validation": tokenized_eval, "test": tokenized_test}

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=lora_model, label_pad_token_id=-100)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Fix shape issues
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = np.squeeze(preds, axis=1)
        if labels.ndim == 3 and labels.shape[1] == 1:
            labels = np.squeeze(labels, axis=1)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
        labels = np.clip(labels, 0, tokenizer.vocab_size - 1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = evaluate.load("rouge").compute(predictions=decoded_preds, references=decoded_labels)
        if isinstance(result["rouge1"], float):
            return {k: v * 100 for k, v in result.items()}
        return {k: v.mid.fmeasure * 100 for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="model_lora_temp",
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        max_grad_norm=0.1,
        eval_steps=5,
        save_steps=5,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        logging_steps=5,
        push_to_hub=True,
        hub_model_id=model_repo_id,
        hub_strategy="end",
        report_to=["tensorboard"]
    )

    trainer = CustomSeq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Start LoRA Fine-tuning on {model_repo_id} ===")
    trainer.train()
    print("=== LoRA Fine-tuning complete ===")

    # Save LoRA weights locally and push to Hugging Face
    trainer.save_model()
    lora_model.save_pretrained(training_args.output_dir)

    final_eval = trainer.evaluate(tokenized_ds["test"])
    print("Trainer Evaluate (test set):", final_eval)

    return lora_model

#############################################################################
#                                  MAIN
#############################################################################

def main():
    # Define dataset and model repository IDs
    dataset_repo_id = "benitoals/my-txt-dataset"
    model_name = "google/mt5-small"
    local_model_repo_id = "benitoals/my-lora"
    hf_model_repo_id = "benitoals/my-lora-hf"
    combined_repo_id = "benitoals/my-lora-local-combined"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)

    # Load pre-existing local dataset
    local_data = load_dataset(dataset_repo_id)
    print("Loaded dataset:", local_data)

    # Filter out short summaries
    local_data = local_data.filter(lambda x: len(x["summary"].split()) >= 50)
    if isinstance(local_data, dict):
        print(f"Filtered dataset sizes - Train: {len(local_data['train'])}, Validation: {len(local_data['validation'])}, Test: {len(local_data['test'])}")
    else:
        print(f"Filtered dataset size: {len(local_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() 
                              else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # Step 1: Baseline (Pretrained model tested on local dataset)
    baseline_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    baseline_rouge = get_rouge_scores(baseline_model, local_data["test"] if isinstance(local_data, dict) else local_data, tokenizer, device)
    print("\n=== Baseline (Pretrained Model) Results ===")
    print(baseline_rouge)

    # Step 2: Train LoRA on local dataset
    local_trained_model = train_lora(baseline_model, local_data, tokenizer, local_model_repo_id)
    local_after_rouge = get_rouge_scores(local_trained_model, local_data["test"] if isinstance(local_data, dict) else local_data, tokenizer, device)
    print("\n=== After LoRA on Local Dataset ===")
    print(local_after_rouge)

    # Step 3: Train on Hugging Face Science dataset
    huggingface_science_repo = "CShorten/ML-ArXiv-Papers"
    hf_data = load_dataset(huggingface_science_repo, split="train").shuffle(seed=42).select(range(1000))
    hf_trained_model = train_lora(
        baseline_model, hf_data, tokenizer, hf_model_repo_id,
        body_key="title", summary_key="abstract"
    )
    hf_on_local_rouge = get_rouge_scores(hf_trained_model, local_data["test"] if isinstance(local_data, dict) else local_data, tokenizer, device)
    print("\n=== After Training on HF Science Dataset ===")
    print(hf_on_local_rouge)

    # Step 4: Fine-tune HF model on local dataset
    # Here we freeze the base (from previous training) and train only the new LoRA adapter
    final_model = train_lora(hf_trained_model, local_data, tokenizer, combined_repo_id, freeze_base=True)
    final_rouge = get_rouge_scores(final_model, local_data["test"] if isinstance(local_data, dict) else local_data, tokenizer, device)
    print("\n=== Final Model (HF + Local) ===")
    print(final_rouge)

    # Print all four results
    print("\n===== All Four Evaluation Results =====")
    print("1) Baseline (Pretrained Model)      =>", baseline_rouge)
    print("2) LoRA on Local Dataset            =>", local_after_rouge)
    print("3) LoRA on HF Dataset               =>", hf_on_local_rouge)
    print("4) Fine-tuned HF + Local Model      =>", final_rouge)

if __name__ == "__main__":
    main()