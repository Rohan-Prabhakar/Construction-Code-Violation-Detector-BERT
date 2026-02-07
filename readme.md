# Building Code Violation Detector

Fine-tuned RoBERTa model for classifying NYC building code violations by category (8 classes) and severity (3 levels).

**See it in action:** [Live Demo](https://construction-code-violation-detector-bert-bkvmmvsnynas4vdk9dic.streamlit.app/)

**Fine-Tuned Model (ViolationBERT):** [HuggingFace](https://huggingface.co/Rohan1103/ViolationBERT)

## Results

| Task | Macro F1 | Accuracy |
|---|---|---|
| Category | 0.8977 | 0.9620 |
| Severity | 0.8637 | 0.8685 |

## Project Structure
```
building-violation-detector/
├── app/                             # Streamlit deployment files
├── checkpoints/                     # Model checkpoints (final_model.pt)
├── data/
│   ├── raw/                         # Raw CSVs from API
│   ├── processed/                   # Cleaned data + tokenized datasets
│   └── splits/                      # train.csv, val.csv, test.csv
├── figures/                         # All generated plots
├── logs/                            # Training logs
├── results/                         # JSON results files
├── data prep and training.ipynb     # Download + Dataset Prep + Model Training
├── evaluation.ipynb                 # Evaluation + Error Analysis + Inference
├── readme.md
└── requirements.txt
```

## Setup
```bash
# Clone the repo
git clone https://github.com/Rohan-Prabhakar/Code-Violation-Detector-BERT-.git
cd Code-Violation-Detector-BERT-

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the notebooks in order:
```
1. data prep and training.ipynb   # Downloads data, preprocesses, trains model (requires GPU)
2. evaluation.ipynb               # Evaluates on test set, error analysis, inference demo
```

For the Streamlit demo:
```bash
pip install streamlit
streamlit run app/app.py
```

## Quick Inference
```python
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch

# Download model from HuggingFace
model_path = hf_hub_download(repo_id="Rohan1103/ViolationBERT", filename="final_model.pt")
ckpt = torch.load(model_path, map_location='cpu')

# Predict
text = "FAILURE TO MAINTAIN BUILDING WALL BRICKS FALLING FROM FACADE"
# Returns: Category=Construction, Severity=HIGH
```

## Dataset

NYC DOB ECB Violations from NYC Open Data

- Source: https://data.cityofnewyork.us/Housing-Development/DOB-ECB-Violations/6bgk-3dad
- 300K records downloaded, 236K after cleaning
- 8 violation categories, 3 severity levels
- Public domain

## Model

- Base: RoBERTa-base (125M params)
- Architecture: Dual classification heads (category + severity)
- Training: Full fine-tuning, AdamW, weighted cross-entropy
- Best config: lr=1e-5, dropout=0.1, batch_size=32, 3 epochs

## Environment

- Python 3.10+
- PyTorch 2.0+ with CUDA
- Trained on Kaggle (T4 GPU)
- Training time: ~4 hours for 3 hyperparameter configs
