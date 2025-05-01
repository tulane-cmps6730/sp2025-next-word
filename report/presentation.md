# Next-Word Prediction While Typing a Message
Zhihe Tu

---

# Background
The ability to predict the next word while typing is crucial for improving user experience in messaging applications.

Accurate next-word prediction can help users formulate their messages faster and correct typos.

Models often struggle with noisy or informal input.

I investigated the task of next-word prediction while a user types a message.

My goal is to assess how well existing language models handle both clean and noisy prompts and improve prediction robustness.

---

# Related Work
Several papers relate to my project:

- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (OpenAI):
  - Introduced GPT-2.
  - Showed that language models can learn tasks without supervised fine-tuning.

- Brown et al. (2020), *Language Models are Few-Shot Learners* (NeurIPS):
  - Demonstrated that larger models like GPT-3 can perform few-shot learning with minimal examples.

Unlike these studies, my project focuses directly on evaluating next-word prediction in real typing scenarios, rather than general multitask learning.

---

# Method
I compare two Transformer-based language models:
- GPT-2
- Unsloth Zephyr(4-bit quantized)

Both models predict the next most likely word based on previous tokens.

To predict the top three next words for each prompt, I apply Top-k sampling, using:
top_k_probs, top_k_indices = torch.topk(probabilities, k=3, dim=-1)

---

# Results

Normal Prompts Accuracy:
- GPT-2: 4/5 = 80.00%
- Unsloth Zephyr: 4/5 = 80.00%

Stress Testing Accuracy:
- GPT-2: 0/3 = 0.00%
- Unsloth Zephyr: 1/3 = 33.33%

---

**Observations**:
- Both models achieve comparable performance on normal English prompts (80% Top-3 accuracy).
- Under stress testing, performance drops sharply:
  - GPT-2 failed completely (0% accuracy).
  - Unsloth Zephyr showed some robustness (33.33% accuracy).

---

**Error Analysis**:
- Systemic biases are observed towards certain structured data. For example, Unsloth Zephyr
  has a clear advantage when handling code-like syntax (e.g., `def calculate_sum(a, b)`).

# Conclusion
This project demonstrates that Unsloth Zephyr and GPT-2 achieve about the same performance on normal English text.

They are vulnerable to stress testing.

We can achieve similar results for typing scenarios with smaller models like Unsloth Zephyr.

Stress testing remains a challenge for deploying reliable typing assistance systems in real-world applications.

---

# Future Work
Next Steps:

- Fine-tune models on noisy, typo-ridden datasets.
- Incorporate error-correction preprocessing techniques.
- Expand from next-word prediction to next-phrase prediction.

Improving robustness against noisy and informal input remains an important research area.

---

# Thank You!
