import evaluate

# Load BERTScore metric
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    """Compute BERTScore for predictions and labels."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels (ensure tokenizer is accessible in the context)
    decoded_preds = [trainer.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [trainer.tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    # Compute BERTScore
    result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    
    # Return all BERTScore metrics
    return {
        "bertscore_precision_mean": result["precision"].mean(),
        "bertscore_recall_mean": result["recall"].mean(),
        "bertscore_f1_mean": result["f1"].mean(),
        "bertscore_precision": result["precision"],
        "bertscore_recall": result["recall"],
        "bertscore_f1": result["f1"]
    }
