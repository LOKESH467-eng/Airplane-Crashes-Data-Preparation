# Aviation SLM V3

A small language model developed for learning and experimenting with aviation-related text data.

## 1. Project Overview

Aviation SLM V3 is a decoder-only Transformer language model trained using aviation-related datasets.

The project focuses on preparing aviation data in a structured format and using it to train a language model that can process and generate aviation-related text.

This repository contains the model architecture, inference code, configuration, and dataset preparation notebook.

## 2. Dataset Source and Preparation

The datasets used in this project were obtained from Kaggle. They were not independently collected by the author.

The downloaded datasets were processed and prepared for model training through the following steps:

- Data cleaning
- Removing duplicate records
- Combining relevant information
- Handling and organizing data
- Converting the data into a structured text format
- Splitting the prepared data into training, validation, and testing sets

The prepared datasets were then used to train Aviation SLM V3.

## 3. Model Architecture

| Feature | Details |
|---|---|
| Model Version | V3 |
| Architecture | Decoder-only Transformer |
| Parameters | Approximately 43.55 million |
| Transformer Layers | 6 |
| Attention Heads | 4 |
| KV Heads | 2 |
| Model Dimension | 256 |
| Feed-Forward Dimension | 768 |
| Vocabulary Size | 151,669 |
| Maximum Sequence Length | 512 |
| Normalization | RMSNorm |
| Position Encoding | RoPE |
| Attention | Grouped Query Attention (GQA) |
| Activation | SwiGLU |
| Embeddings | Tied Input and Output Embeddings |

## 4. Repository Structure

```text
Airplane-Crashes-Data-Preparation/
│
├── model/
│   ├── model.py
│   ├── inference.py
│   └── config.json
│
├── dataset_preparation/
│   ├── Airplane_Crashes.ipynb
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

The trained model checkpoint and tokenizer files are hosted separately on Hugging Face.

## 5. Installation

Clone the repository:

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

## 6. Running the Model

To run the model:

1. Download the V3 model checkpoint from the Hugging Face repository.
2. Download the required tokenizer files.
3. Place the files in the locations expected by the inference script.
4. Check the file paths in `model/inference.py`.
5. Run the inference script:

```bash
python model/inference.py
```

Refer to the inference code for the expected file paths and input format.

## 7. Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Jupyter Notebook

## 8. Limitations

This project is developed for educational and research purposes.

The model may generate incorrect, incomplete, or repetitive information. Its responses should not be considered verified aviation facts.

It must not be used for real flight operations, aircraft maintenance, or aviation safety decisions.

## 9. Author

**Lokesh Addepalli**