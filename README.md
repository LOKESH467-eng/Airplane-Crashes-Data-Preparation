# Aviation SLM V3

A compact decoder-only Transformer language model trained on aviation-related data.

## 1. Project Overview

Aviation SLM V3 is a domain-specific language model developed to work with aviation-related text data.

The project involves collecting aviation data, preparing and processing records, tokenizing the data, and training a custom Transformer-based language model.

This repository contains the model architecture, inference code, configuration-related resources, and dataset preparation notebook.

## 2. Data Source

The original aviation data used in this project was collected from **Kaggle**.

The collected data was cleaned, processed, and prepared before being used for model training. The records were converted into a tokenized format using the project's tokenizer.

The final dataset was divided into training, validation, and testing sets.

## 3. Model Architecture

Aviation SLM V3 uses a decoder-only Transformer architecture.

| Component | Configuration |
|---|---|
| Model type | Decoder-only Transformer |
| Transformer layers | 6 |
| Model dimension | 256 |
| Attention heads | 4 |
| Key-value heads | 2 |
| Feed-forward dimension | 768 |
| Vocabulary size | 151,669 |
| Maximum sequence length | 512 |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Attention mechanism | Grouped Query Attention |
| Feed-forward network | SwiGLU |
| Parameters | 43,549,184 |

The model also uses tied input and output embeddings.

## 4. Dataset Details

The V3 training dataset contains **1,533,723 records**.

The dataset is divided into three splits:

| Dataset Split | Number of Records |
|---|---:|
| Training | 1,226,943 |
| Validation | 153,367 |
| Testing | 153,413 |
| **Total** | **1,533,723** |

### Tokenization Details

- Total tokens: 422,889,437
- Tokenizer vocabulary size: 151,669
- Tokenizer: Qwen2-compatible tokenizer
- EOS token: `<|im_end|>`
- EOS token ID: 151645
- Data type: uint32
- One EOS separator is appended after every record.
- No truncation was applied during tokenized dataset preparation.

### Dataset Files

The tokenized V3 dataset is available in the `tokenized_v3/` folder of the Hugging Face dataset repository.

Files:
- `train.parquet`
- `validation.parquet`
- `test.parquet`

Dataset repository:

[Airplane Crashes Dataset on Hugging Face](https://huggingface.co/datasets/lokesh23112007/Airplane_crashes_dataset)

## 5. Model Files

The trained V3 model and tokenizer are hosted separately on Hugging Face.

Model repository:

[Aviation SLM on Hugging Face](https://huggingface.co/lokesh23112007/aviation-slm)

The repository contains:
- `aviation_slm_v3.pt` - Trained model checkpoint
- `config.json` - Model configuration
- `tokenizer/` - Tokenizer files

The tokenizer folder contains:
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`

## 6. GitHub Repository Structure

```text
Airplane-Crashes-Data-Preparation/
│
├── dataset_preparation/
│   ├── Airplane_Crashes.ipynb
│   └── README.md
│
├── model/
│   ├── model.py
│   └── inference.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The model checkpoint and tokenizer files are hosted on Hugging Face rather than stored directly in this GitHub repository.

## 7. Installation

Clone the GitHub repository:

```bash
git clone https://github.com/LOKESH467-eng/Airplane-Crashes-Data-Preparation.git
```

Move into the project folder:

```bash
cd Airplane-Crashes-Data-Preparation
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 8. Running Inference

To run the model:

1. Download the V3 checkpoint and configuration from the Hugging Face model repository.
2. Download or access the tokenizer files.
3. Place the files in the locations expected by the inference script.
4. Run the inference script:

```bash
python model/inference.py
```

Refer to `inference.py` for the expected file paths and loading procedure.

## 9. Training Process

The V3 training process involved:

1. Collecting aviation-related data from Kaggle.
2. Cleaning and preparing the collected data.
3. Creating training-ready aviation text records.
4. Tokenizing the records using the project tokenizer.
5. Dividing the tokenized data into training, validation, and testing sets.
6. Training the decoder-only Transformer model.
7. Saving the trained model checkpoint and configuration.
8. Verifying the tokenized dataset splits.

## 10. Limitations

- Aviation SLM V3 is an experimental domain-specific language model.
- Its generated responses may contain incorrect, incomplete, or unrelated information.
- It is not a production-ready aviation system.
- It must not be used for flight operations, aircraft maintenance, accident investigations, or other safety-critical decisions.
- Outputs should be independently checked before being used for research or educational purposes.

## 11. Project Resources

| Resource | Link |
|---|---|
| GitHub Repository | [Airplane Crashes Data Preparation](https://github.com/LOKESH467-eng/Airplane-Crashes-Data-Preparation) |
| Hugging Face Model | [Aviation SLM](https://huggingface.co/lokesh23112007/aviation-slm) |
| Hugging Face Dataset | [Airplane Crashes Dataset](https://huggingface.co/datasets/lokesh23112007/Airplane_crashes_dataset) |
