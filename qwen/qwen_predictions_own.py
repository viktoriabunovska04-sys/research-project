"""Qwen hate speech classification on its own default definition of hate speech..

This script loads the hate speech dataset, prepares a prompt-based binary
classification request for Qwen, generates only the token "0" or "1", and
writes the predicted labels back to a CSV for later analysis.
"""

import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# file reading
path_general = "data/hatecheck_with_dominant.xlsx"
dataset = pd.read_excel(path_general, engine="openpyxl")


# exctract true labels
label_map = {"hateful": 1, 
             "non-hateful": 0}
dataset["label_gold"] = dataset["label_gold"].map(label_map)


model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  
    device_map="cuda"
)
model.eval()
device = model.device


def build_prompt(sample):
    """Build a chat-style prompt for Qwen to classify a single text sample.

    The prompt asks the model to reply with exactly one token: "1" for hate
    speech or "0" for not hate speech. The classifier is constrained to this
    binary output using prefix token filtering during generation.

    Args:
        sample (str): The text to classify.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a hate speech classifier. "
                "You must respond with ONLY the digit 1 or 0. No other output is allowed.\n"
                "1 = hate speech\n"
                "0 = not hate speech\n\n"
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
def classify_batch(samples, num_runs=1, batch_size=8):
    """Classify a batch of text samples using the Qwen model.

    Samples are converted into chat prompts, tokenized, and fed to the model in
    batches. Generation is constrained so that the model can only produce the
    binary label tokens "0" or "1". The function supports repeated runs for
    experimental stability checks.

    Args:
        samples (Iterable[str]): Text examples to classify.
        num_runs (int, optional): Number of repeated classification runs.
            Defaults to 1.
        batch_size (int, optional): Number of samples to process per batch.
            Defaults to 8.

    Returns:
        list[list[int]]: A list of prediction lists, one per run.

    Raises:
        ValueError: If the model generates a token outside the allowed set.
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
own_csv.to_csv("results/predictions_qwen/own_csv_qwen.csv", index=False)
