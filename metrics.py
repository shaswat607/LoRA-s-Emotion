import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = [trainer.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
    decoded_labels = [trainer.tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure
    }
