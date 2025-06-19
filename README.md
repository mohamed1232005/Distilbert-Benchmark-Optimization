# 🧪 DistilBERT Benchmarking, Optimization & Fine-Tuning on WikiText & GSM8K

---

## 📘 Project Overview

This project focuses on the **benchmarking, optimization, and fine-tuning** of the **DistilBERT** language model using two benchmark datasets: **WikiText-2** and **GSM8K**. We aim to compare **quality** and **performance** across three model variants:

1. **Base Pretrained DistilBERT**
2. **Optimized Model using 8-bit Quantization**
3. **Fine-Tuned Model on WikiText**

We provide an end-to-end pipeline implemented in **Google Colab** with the following deliverables:
- Metric evaluations: **Loss**, **Perplexity**
- Performance metrics: **Throughput (samples/sec)**
- Optimization using **quantization**
- Fine-tuning with **Hugging Face Trainer API**
- Visual performance comparisons

---

## 📊 Core Concepts

### 🔍 What is Benchmarking?

> Benchmarking is the process of **evaluating the performance** of a machine learning model under various constraints and comparing it with alternative models or configurations.

We benchmark in **two dimensions**:
1. **Accuracy-Oriented Metrics**
   - **Loss (Cross-Entropy Loss)**: Indicates how far off the model's predictions are.
   - **Perplexity**: Measures model confidence in predicting tokens.

2. **Performance-Oriented Metrics**
   - **Throughput**: Speed of processing samples (in samples/sec).

### ⚙️ What is Throughput?

Throughput is a **runtime metric** representing the number of samples the model can process per second. Higher throughput is critical for real-time or large-scale applications.

\[
\text{Throughput} = \frac{\text{Number of processed samples}}{\text{Time taken (in seconds)}}
\]

Measured by:
- Inference loop wrapped with `time.time()` before and after.
- Evaluated per dataset and per model version.

### ⚡ What is Quantization?

Quantization reduces the **precision** of model parameters from 32-bit floats (FP32) to **8-bit integers (INT8)** using **bitsandbytes**. This leads to:
- Smaller memory footprint
- Lower computation cost
- Faster inference speed (improved throughput)

Quantization is **post-training**, so no retraining is needed before use.

---

## 📚 Datasets

### 1. **WikiText-2**
- Source: Wikipedia articles.
- Use: Standard for evaluating language modeling.

### 2. **GSM8K**
- Source: Grade-school level math word problems.
- Use: Evaluates structured reasoning and math understanding.

```python
from datasets import load_dataset

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
gsm8k = load_dataset("gsm8k", "main")
```

---

## 🤖 Model: DistilBERT

**DistilBERT** is a lighter, faster version of BERT developed by Hugging Face via **knowledge distillation**.

| Model     | Parameters | Speed Gain | Size Reduction | Accuracy Retention |
|-----------|------------|------------|----------------|---------------------|
| BERT-base | 110M       | —          | —              | —                   |
| DistilBERT| ~66M       | ~60%       | ~40%           | ~97%                |

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
```

---

## 🛠️ Technologies & Libraries

| Library        | Purpose                                      | Link                                           |
|----------------|----------------------------------------------|------------------------------------------------|
| `transformers` | Load and fine-tune DistilBERT                | https://huggingface.co/transformers            |
| `datasets`     | Load WikiText and GSM8K datasets             | https://huggingface.co/docs/datasets           |
| `evaluate`     | Compute metrics like perplexity              | https://huggingface.co/docs/evaluate           |
| `bitsandbytes` | Apply 8-bit quantization for optimization    | https://github.com/TimDettmers/bitsandbytes    |
| `torch`        | Core PyTorch library for ML training         | https://pytorch.org/                           |
| `matplotlib`   | Plot model comparison results                | https://matplotlib.org/                        |
| `accelerate`   | Speed up inference with GPU optimizations    | https://github.com/huggingface/accelerate      |

---

## 📈 Experimental Setup

### 🔹 Hardware Used

| Component        | Description              |
|------------------|--------------------------|
| Runtime Platform | Google Colab Pro         |
| GPU              | NVIDIA Tesla T4 (16 GB)  |
| CPU              | Intel Xeon               |
| RAM              | ~25 GB                   |
| Python           | 3.10                     |
| PyTorch          | 2.x                      |

---

## 🔬 Results and Analysis

### 📊 Benchmark Results (All Stages)

| Stage            | Dataset   | Loss   | Perplexity | Throughput (samples/sec) |
|------------------|-----------|--------|------------|---------------------------|
| Base             | WikiText  | 9.84   | 18,764.7   | 215.54                    |
| Base             | GSM8K     | 7.18   | 1,313.57   | 218.46                    |
| Quantized        | WikiText  | 10.12  | 24,679.8   | 251.56                    |
| Quantized        | GSM8K     | 7.30   | 1,478.95   | 260.46                    |
| Fine-Tuned       | WikiText  | 8.95   | 7,703.15   | 248.12                    |
| Fine-Tuned       | GSM8K     | 6.83   | 927.16     | 262.08                    |

---

## 🧪 Fine-Tuning Details

We fine-tuned DistilBERT using the Hugging Face `Trainer` API.

### Parameters:
| Parameter       | Value            |
|------------------|------------------|
| Epochs           | 1                |
| Batch Size       | 16               |
| Data Collator    | MLM (15% mask)   |
| Evaluation       | After each epoch |

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_wikitext,
    eval_dataset=tokenized_gsm8k,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
```

---

## 📉 Visualization

```python
import matplotlib.pyplot as plt

models = ["Base", "Quantized", "Fine-Tuned"]
throughput = [215.54, 251.56, 248.12]
perplexity = [18764.7, 24679.8, 7703.15]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(models, throughput, 'g-')
ax2.plot(models, perplexity, 'b-')

ax1.set_xlabel('Model Version')
ax1.set_ylabel('Throughput (samples/sec)', color='g')
ax2.set_ylabel('Perplexity', color='b')
plt.title("Throughput vs Perplexity Across Model Versions")
plt.show()
```

---

## ✅ Key Takeaways

| Aspect             | Observation                                                                 |
|--------------------|------------------------------------------------------------------------------|
| Quantization       | Increases throughput significantly (~15–20%)                                |
| Accuracy Loss      | Small drop in performance after quantization (acceptable for real-time use) |
| Fine-tuning        | Recovers accuracy while keeping most of the speed gain                      |
| DistilBERT Utility | Excellent balance between model size, accuracy, and speed                   |

---

## 📦 Installation

Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

### `requirements.txt` content:
```txt
transformers
datasets
evaluate
bitsandbytes
torch
accelerate
matplotlib
```

---


