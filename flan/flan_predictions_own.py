import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import pandas as pd

"""FLAN own default hate speech classification script.

This module uses a Flan-T5 model to classify text as hate speech (1) or not hate speech (0)
with its own default definition, and saves results to CSV files. 
Make sure to configure access to Hugging Face, depending on the environment in which you run the script. 
"""

# file reading
path = "data/hatecheck_with_dominant.xlsx"  # Replace with your dataset path
dataset = pd.read_excel(path, engine="openpyxl")

# map true labels to 1 or 0 for each definition
label_map = {"hateful": 1, 
             "non-hateful": 0}
dataset["label_gold"] = dataset["label_gold"].map(label_map)


model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=torch.float16,  
    device_map="auto"
)
model.eval()
device = model.device


def build_prompt(sample):
    """Construct a prompt for hate speech classification using model's default definition.

    Args:
        sample: The text sample to classify.
    """

    return f"""
    Classify if the following text is considered hate speech or not hate speech:
    Text: {sample}
    Please answer ONLY with 1 if it is hate speech or 0 if it is not hate speech.
    """

zero_id = tokenizer.convert_tokens_to_ids("▁0") # Flan-T5 uses SentencePiece 
one_id = tokenizer.convert_tokens_to_ids("▁1") # Fallback if the above gives <unk> 
if zero_id == tokenizer.unk_token_id: 
    zero_id = tokenizer.encode("0", add_special_tokens=False)[0] 
    one_id = tokenizer.encode("1", add_special_tokens=False)[0] 

def allowed_tokens():
    """Return the token ids permitted for model generation.

    The FLAN model is constrained to emit only the tokens representing
    "0" or "1" so that outputs can be interpreted as binary labels.

    Returns:
        list[int]: A list containing the allowed token ids for zero and one.
    """ 
    return [zero_id, one_id]

@torch.inference_mode()
def classify_batch(samples, num_runs=1, batch_size=8):
    """Classify a batch of text samples with the FLAN model.

    Args:
        samples (Iterable[str]): The text examples to classify.
        num_runs (int, optional): Number of repeated inference runs to return.
            Defaults to 1, as we use a deterministic setup now.
        batch_size (int, optional): Number of samples processed per model batch.
            Defaults to 8.

    Returns:
        list[list[int]]: A list of runs, each containing a list of binary
        predictions (1 for hate speech, 0 for non-hate speech).

    Raises:
        ValueError: If the model emits a token outside the allowed set.
    """
    results = []

    for run in range(num_runs):
        run_predictions = []

        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i:i + batch_size]
            prompts = [build_prompt(sample=s) for s in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                prefix_allowed_tokens_fn=allowed_tokens
            )
            predictions=[]
            for tid in outputs[:, 1].tolist():
                if tid == one_id:
                    predictions.append(1)
                elif tid == zero_id:
                    predictions.append(0)
                else:
                    raise ValueError(f"Unexpected token id: {tid}")
            run_predictions.extend(predictions)

        results.append(run_predictions)

    return results

samples = dataset["test_case"]
results = classify_batch(samples)
own_csv = pd.DataFrame({
    "functionality": dataset["functionality"],
    "test_case": samples,
    "target_type": dataset["target_type"],
    "dominance": dataset["dominance"],
    "explicit_reference": dataset["explicit_ref"],
    "consequences": dataset["incites"],
    "group_insult": dataset["group_insult"],
    "in_group": dataset["in_group"],
    "prediction": results[0],
    "label_gold": dataset["label_gold"]
})
own_csv.to_csv("results/predictions_flan/own_csv_flan.csv", index=False)
