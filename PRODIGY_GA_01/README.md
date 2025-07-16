# Task-01 ProDigy Infotech: Text Generation with GPT-2 🍭🌱

## Overview
In this task, the goal is to train a model to generate coherent and contextually relevant text based on a given prompt. Starting with GPT-2, a transformer model developed by OpenAI, the focus is on fine-tuning the model on a custom dataset to create text that mimics the style and structure of the data. 

### References
- [How to Generate Text with GPT-2](https://huggingface.co/blog/how-to-generate)
- [Colab Notebook for GPT-2 Fine-Tuning](https://colab.research.google.com/drive/15qBZx5y9rdaQSyWpsreMDnTiZ5IlN0zD?usp=sharing)

<img src="task1.png" >

### Key Learning Goals:
“For Generative AI Interns, focus on learning the topics and understanding the theory in as much detail as possible. The output doesn’t matter if you don’t grasp the theory behind these complex topics.”

## Steps Completed

### 1. Environment Setup
- Installed the necessary libraries: `transformers`, `torch`, and `datasets`.
- Configured the environment using Jupyter Notebook/Colab for smooth execution.

### 2. Dataset Preparation
- Custom dataset collected and preprocessed for fine-tuning.
- Tokenized the text data and converted it to the format required for GPT-2.

### 3. Fine-Tuning the GPT-2 Model
- Fine-tuned GPT-2 using a custom dataset.
- Adjusted hyperparameters such as learning rate, batch size, and epochs.
- Monitored model performance using loss during training.

### 4. Testing and Evaluation

#### Loading the Fine-Tuned Model:
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('path_to_saved_model_directory')
tokenizer = GPT2Tokenizer.from_pretrained('path_to_saved_model_directory')
```

#### Text Generation:
```python
input_ids = tokenizer.encode("Test prompt", return_tensors='pt')
output = model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

#### Testing with Different Decoding Methods:
- **Greedy Search**
- **Beam Search**
- **Top-K Sampling**
- **Top-p Sampling**

#### Evaluation Criteria:
- **Coherence**: Logical flow in text generation.
- **Relevance**: On-topic and reflective of training data.
- **Diversity**: Creativity and avoidance of repetition.

#### Perplexity Calculation:
```python
def calculate_perplexity(model, input_text):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()

perplexity = calculate_perplexity(model, "This is a test sentence.")
print(f"Perplexity: {perplexity}")
```

### 5. Refinement
- Further tuning done based on text generation results.
- Experimented with hyperparameters like `temperature`, `top_k`, and `top_p` for better diversity and creativity.

## Next Steps
- Deploy the model in a small application or interface.
- Continuously test and improve the model based on feedback.
