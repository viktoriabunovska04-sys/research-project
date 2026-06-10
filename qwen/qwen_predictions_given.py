"""Run Qwen-based hate speech classification and save predictions.

This script loads multiple hate speech datasets, applies definition-conditioned
prompting using the Qwen model, and stores binary predictions in CSV files.
The model is constrained to output only the digits 0 or 1, where 1 indicates
hate speech and 0 indicates non-hate speech.
"""

from tqdm.auto import tqdm
import pandas as pd
from transformers import GenerationConfig
import torch
from src.definitions import DEFINITIONS
from transformers import AutoTokenizer, AutoModelForCausalLM


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


# exctract true labels
label_map = {"hateful": 1, "non-hateful": 0}
for ds in [
    dataset_bulgaria,
    dataset_croatia,
    dataset_meta,
    dataset_reddit,
    dataset_theoretical
]:
    ds["def_label"] = ds["def_label"].map(label_map)


model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  
    device_map="cuda"
)

model.eval()
device = model.device


def build_prompt(definition, sample):
    """Build a chat-style prompt for Qwen hate speech classification.

    Args:
        definition (str): The definition used to guide the model.
        sample (str): The text sample to classify.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a hate speech classifier. "
                "You must respond with ONLY the digit 1 or 0. No other output is allowed.\n"
                "1 = hate speech\n"
                "0 = not hate speech\n\n"
                f"Use this definition to guide your classification:\n{definition}"
            )
        },
        {
            "role": "user",
            "content": f'Text: "{sample}"\nAnswer:'
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


zero_id = tokenizer.encode("0", add_special_tokens=False)[0]
one_id  = tokenizer.encode("1", add_special_tokens=False)[0]

def allowed_tokens():
    return [zero_id, one_id]

gen_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=1,
)

@torch.inference_mode()
def classify_batch(samples, definition, num_runs=1, batch_size=8):
    """Classify a list of samples using the Qwen model and a definition prompt.

    Args:
        samples (Sequence[str]): Text samples to classify.
        definition (str): The guiding definition used in the prompt.
        num_runs (int, optional): Number of repeated inference runs. Defaults to 1.
        batch_size (int, optional): Number of samples processed per batch. Defaults to 8.
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

            input_length = inputs["input_ids"].shape[1]  # track where input ends

            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                prefix_allowed_tokens_fn=allowed_tokens
            )

            predictions = []
            for tid in outputs[:, input_length].tolist():
                if tid == one_id:
                    predictions.append(1)
                elif tid == zero_id:
                    predictions.append(0)
                else:
                    raise ValueError(f"Unexpected token id: {tid}")
            run_predictions.extend(predictions)
        results.append(run_predictions)
    return results



# test samples
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
    "results/predictions_qwen/bulgaria_results_qwen.csv"
)

save_results(
    dataset_croatia,
    results_croatia[0],
    "results/predictions_qwen/croatia_results_qwen.csv"
)

save_results(
    dataset_meta,
    results_meta[0],
    "results/predictions_qwen/meta_results_qwen.csv"
)

save_results(
    dataset_reddit,
    results_reddit[0],
    "results/predictions_qwen/reddit_results_qwen.csv"
)

save_results(
    dataset_theoretical,
    results_theoretical_incl[0],
    "results/predictions_qwen/theoretical_incl_results_qwen.csv"
)

save_results(
    dataset_theoretical,
    results_theoretical_incl_excl[0],
    "results/predictions_qwen/theoretical_incl_excl_results_qwen.csv"
)
