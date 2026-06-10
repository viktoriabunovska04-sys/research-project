import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.definitions import DEFINITIONS

"""FLAN hate speech classification script.

This module loads hate speech datasets, builds definition-conditioned prompts,
uses a Flan-T5 model to classify text as hate speech (1) or not hate speech (0),
and saves results to CSV files. Make sure to configure access to Hugging Face, 
depending on where you run the script. 
"""

# file reading
path_general = "data/hatecheck_with_dominant.xlsx"
path_bulgaria = "data/hatecheck_with_dominant_bulgaria.xlsx"
path_croatia = "data/hatecheck_with_dominant_croatia.xlsx"
path_meta = "data/hatecheck_with_dominant_meta.xlsx"
path_reddit = "data/hatecheck_with_dominant_reddit.xlsx"
path_theoretical = "data/hatecheck_with_dominant_theoretical.xlsx" # only this since both theoretically-created have the same meaning and labels

dataset_general = pd.read_excel(path_general, engine="openpyxl")
dataset_bulgaria = pd.read_excel(path_bulgaria, engine="openpyxl")
dataset_croatia = pd.read_excel(path_croatia, engine="openpyxl")
dataset_meta = pd.read_excel(path_meta, engine="openpyxl")
dataset_reddit = pd.read_excel(path_reddit, engine="openpyxl")
dataset_theoretical = pd.read_excel(path_theoretical, engine="openpyxl")

# map true labels to 1 or 0 for each definition
label_map = {"hateful": 1, "non-hateful": 0}
for ds in [
    dataset_bulgaria,
    dataset_croatia,
    dataset_meta,
    dataset_reddit,
    dataset_theoretical
]:
    ds["def_label"] = ds["def_label"].map(label_map)


model_name = "google/flan-t5-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=torch.float16,  
    device_map="auto"
)
model.eval()
device = model.device


def build_prompt(definition, sample):
    """Construct a prompt for definition-conditioned hate speech classification.

    Args:
        definition: The hate speech definition text to include in the prompt.
        sample: The text sample to classify.
    """
    return f"""Given the following definition of Hate Speech: \"{definition}\"
            Classify if the following text is considered hate speech or not hate speech:
            Text: \"{sample}\"
            Please answer ONLY with 1 if it is hate speech or 0 if it is not hate speech."""

zero_id = tokenizer.convert_tokens_to_ids("▁0") # Flan-T5 uses SentencePiece 
one_id = tokenizer.convert_tokens_to_ids("▁1") # Fallback if the above gives <unk> 
if zero_id == tokenizer.unk_token_id: 
    zero_id = tokenizer.encode("0", add_special_tokens=False)[0] 
    one_id = tokenizer.encode("1", add_special_tokens=False)[0] 

def allowed_tokens():
    """Return the token ids permitted for model generation.

    The FLAN model is constrained to emit only the tokens representing
    "0" or "1" so that outputs can be interpreted as binary labels.
    """
    return [zero_id, one_id]

@torch.inference_mode()
def classify_batch(samples, definition, num_runs=1, batch_size=8):
    """Classify a batch of text samples with the FLAN model.

    Args:
        samples (Iterable[str]): The text examples to classify.
        definition (str): The hate speech definition used in the prompt.
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
            prompts = [build_prompt(definition=definition, sample=s) for s in batch]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            
            outputs = model.generate(
                **inputs,
                max_new_tokens=1, # restricts token size
                do_sample=False, # ensures deterministic behavior
                prefix_allowed_tokens_fn=allowed_tokens #restricts tokens
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


samples = dataset_general["test_case"]

results_bulgaria = classify_batch(samples, DEFINITIONS["Bulgaria"])
results_croatia = classify_batch(samples, DEFINITIONS["Croatia"])
results_meta = classify_batch(samples, DEFINITIONS["Meta"])
results_reddit = classify_batch(samples, DEFINITIONS["Reddit"])
results_theoretical_incl = classify_batch(samples, DEFINITIONS["Theoretic Inclusion"])
results_theoretical_incl_excl = classify_batch(samples, DEFINITIONS["Theoretic Inclusion + Exclusion"])

#save the results to csv files
def save_results(dataset, predictions, filename):
    """Save classification predictions and dataset metadata to CSV.

    Args:
        dataset (pandas.DataFrame): The dataset containing metadata fields.
        predictions (list[int]): Model predictions aligned with dataset rows.
        filename (str): Output CSV path.
    """
    df = pd.DataFrame({
        "functionality": dataset["functionality"],
        "test_case": dataset["test_case"],
        "target_type": dataset["target_type"],
        "dominance": dataset["dominance"],
        "explicit_reference": dataset["explicit_ref"],
        "consequences": dataset["incites"],
        "group_insult": dataset["group_insult"],
        "in_group": dataset["in_group"],
        "prediction": predictions,
        "true_label": dataset["def_label"]
    })

    df.to_csv(filename, index=False)


save_results(
    dataset_bulgaria,
    results_bulgaria[0],
    "results/predictions_flan/bulgaria_results_flan.csv"
)

save_results(
    dataset_croatia,
    results_croatia[0],
    "results/predictions_flan/croatia_results_flan.csv"
)

save_results(
    dataset_meta,
    results_meta[0],
    "results/predictions_flan/meta_results_flan.csv"
)

save_results(
    dataset_reddit,
    results_reddit[0],
    "results/predictions_flan/reddit_results_flan.csv"
)

save_results(
    dataset_theoretical,
    results_theoretical_incl[0],
    "results/predictions_flan/theoretical_incl_results_flan.csv"
)

save_results(
    dataset_theoretical,
    results_theoretical_incl_excl[0],
    "results/predictions_flan/theoretical_incl_excl_results_flan.csv"
)