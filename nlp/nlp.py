!pip install -U bitsandbytes unsloth
!pip install transformers matplotlib
!pip install huggingface_hub[hf_xet]
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unsloth import FastLanguageModel, PatchDPOTrainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 Model and Tokenizer
def load_gpt2():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Load Unsloth Model and Tokenizer
def load_unsloth():
    PatchDPOTrainer()
    max_seq_length = 1024
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/zephyr-sft-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    device = model.device
    return model, tokenizer, device

# Prediction Function
def predict_next_words(prompt, model, tokenizer, device, top_k=3):
    if not prompt.strip():
        return [("No input", 0.0)]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]
    probabilities = F.softmax(next_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
    predictions = []
    for i in range(top_k):
        token_id = top_k_indices[0, i].item()
        token = tokenizer.decode([token_id]).strip()
        prob = top_k_probs[0, i].item()
        predictions.append((token, prob))
    return predictions

# Plotting Normal and Stress Testing Accuracies Separately
def plot_accuracy(normal_accuracies, stress_accuracies):
    plt.figure(figsize=(8, 6))
    models = ['GPT-2', 'Unsloth']
    plt.bar(models, normal_accuracies, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Normal Prompt Accuracy Comparison', fontsize=14)
    plt.grid(axis='y')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(models, stress_accuracies, color=['cyan', 'lime'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Stress Testing Accuracy Comparison', fontsize=14)
    plt.grid(axis='y')
    plt.show()

# Main Testing Function
def test_models_on_prompts(gpt2_model, gpt2_tokenizer, gpt2_device,
                           unsloth_model, unsloth_tokenizer, unsloth_device,
                           prompts, ground_truths, top_k=3):
    gpt2_correct = 0
    unsloth_correct = 0

    for prompt, truth in zip(prompts, ground_truths):
        gpt2_preds = predict_next_words(prompt, gpt2_model, gpt2_tokenizer, gpt2_device, top_k)
        unsloth_preds = predict_next_words(prompt, unsloth_model, unsloth_tokenizer, unsloth_device, top_k)

        gpt2_predicted_words = [word.lower() for word, _ in gpt2_preds]
        unsloth_predicted_words = [word.lower() for word, _ in unsloth_preds]

        if truth.lower() in gpt2_predicted_words:
            gpt2_correct += 1
        if truth.lower() in unsloth_predicted_words:
            unsloth_correct += 1

    total = len(prompts)
    gpt2_accuracy_normal = gpt2_correct / total
    unsloth_accuracy_normal = unsloth_correct / total

    # Stress Testing
    stress_prompts = ["segdgegedd","Heo wrld","def calculate_sum(a, b)"]
    stress_ground_truths = ["", "world", ":"]
    gpt2_stress_correct = 0
    unsloth_stress_correct = 0

    print("\n===== Stress Testing Results =====")
    for prompt, truth in zip(stress_prompts, stress_ground_truths):
        print(f"\nStress Test Prompt: '{prompt}'")
        gpt2_preds = predict_next_words(prompt, gpt2_model, gpt2_tokenizer, gpt2_device, top_k)
        unsloth_preds = predict_next_words(prompt, unsloth_model, unsloth_tokenizer, unsloth_device, top_k)
        print("GPT-2 Predictions:", gpt2_preds)
        print("Unsloth Predictions:", unsloth_preds)

        gpt2_predicted_words = [word.lower() for word, _ in gpt2_preds]
        unsloth_predicted_words = [word.lower() for word, _ in unsloth_preds]

        if truth and truth.lower() in gpt2_predicted_words:
            gpt2_stress_correct += 1
        if truth and truth.lower() in unsloth_predicted_words:
            unsloth_stress_correct += 1

    total_stress = len(stress_prompts)
    gpt2_accuracy_stress = gpt2_stress_correct / total_stress
    unsloth_accuracy_stress = unsloth_stress_correct / total_stress

    print("\n===== Accuracy Results =====")
    print(f"GPT-2 Normal Accuracy: {gpt2_correct}/{total} = {gpt2_accuracy_normal:.2%}")
    print(f"Unsloth Normal Accuracy: {unsloth_correct}/{total} = {unsloth_accuracy_normal:.2%}")
    print(f"GPT-2 Stress Accuracy: {gpt2_stress_correct}/{total_stress} = {gpt2_accuracy_stress:.2%}")
    print(f"Unsloth Stress Accuracy: {unsloth_stress_correct}/{total_stress} = {unsloth_accuracy_stress:.2%}")

    plot_accuracy([gpt2_accuracy_normal, unsloth_accuracy_normal], [gpt2_accuracy_stress, unsloth_accuracy_stress])

# Main Script
if __name__ == "__main__":
    # Load models
    gpt2_model, gpt2_tokenizer, gpt2_device = load_gpt2()
    unsloth_model, unsloth_tokenizer, unsloth_device = load_unsloth()

    # Define prompts and their ground truths
    prompts = [
        "The quick brown fox",
        "How are",
        "What are the cause of global",
        "Climate change continues to impact",
        "Do you want a cup of"
    ]

    ground_truths = [
        "jumped",
        "you",
        "warming",
        "the",
        "tea"
    ]

    # Run the tests
    test_models_on_prompts(
        gpt2_model, gpt2_tokenizer, gpt2_device,
        unsloth_model, unsloth_tokenizer, unsloth_device,
        prompts,
        ground_truths
    )# -*- coding: utf-8 -*-

"""Main module."""
