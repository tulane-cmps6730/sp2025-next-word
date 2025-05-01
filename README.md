# Project Summary
This project explores next-word prediction while typing a message.  
I compare two transformer-based models, GPT-2 and Unsloth Zephyr (4-bit quantized), evaluating their performance on both normal English prompts and stress testing.

---

# Goals
- Investigate how well different language models can predict the next word while typing.
- Evaluate model performance under both normal and stress-testing conditions.
- Identify systemic biases and weaknesses (e.g., struggles with typos, code-like syntax).
- Recommend future improvements for real-world typing assistance systems.

---

# Methods
- **Models**:
  - GPT-2 (baseline)
  - Unsloth Zephyr SFT (4-bit quantized)
- **Techniques**:
  - Top-k Sampling to select the top 3 next-word predictions.
  - Standard normal prompts (e.g., "The quick brown fox...").
  - Stress testing with typos, random tokens, and code-like prompts.
- **Evaluation**:
  - Top-3 Accuracy on both normal and noisy prompts.
  - Visual comparison using bar charts.

---

# Results

| Test Type               | GPT-2 Accuracy | Unsloth Zephyr Accuracy |
|--------------------------|----------------|-------------------------|
| Normal Prompts           | 80%            | 80%                     |
| Stress Testing Prompts   | 0%             | 33%                     |

- **Normal Prompts**: Both models performed similarly, achieving 80% accuracy.
- **Stress Testing**: Unsloth Zephyr showed higher robustness against noisy input compared to GPT-2.

---

# Conclusion
- Both models achieve comparable results on normal English text.
- Unsloth Zephyr outperforms GPT-2 in handling typos and random input (stress testing).
- **Future directions**:
  - Fine-tuning on typo-corrected or noisy datasets.
  - Adding spell-correction preprocessing.
  - Expanding to next-phrase prediction instead of next-word only.

---

# Takeaway
Smaller, efficient models like **Unsloth Zephyr** can achieve competitive results in typing scenarios while offering better robustness against noisy real-world input.

---
# Screenshot

![image](https://github.com/user-attachments/assets/cd446339-8659-42b4-bf06-88a5a3f0567f)
![image](https://github.com/user-attachments/assets/bdfdf1bf-1d5e-4ec3-85a2-af0bdaf33c8f)




