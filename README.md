# POS-Tagging

This project implements Part-of-Speech (POS) tagging using a logistic regression model. The goal of POS tagging is to label words in a sentence with their appropriate grammatical tags (such as noun, verb, adjective, etc.). The code follows a pipeline of preprocessing text data, training a logistic regression model, performing hyperparameter tuning, and making predictions.

---

## Table of Contents

- [POS-Tagging](#pos-tagging)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [File Structure](#file-structure)
  - [Usage](#usage)

---

## Overview

This project employs a logistic regression model for POS tagging. It takes a labeled dataset in CoNLL format, tokenizes sentences, generates features for each token, trains the model, performs hyperparameter tuning, and evaluates the model on a development set. The model's performance is measured by its accuracy on the development set. After training, the model can be used for POS tag prediction on unseen text.

---

## Requirements

To run this project, you need the following Python libraries:

- `torch` (PyTorch)
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## File Structure

The project directory contains the following files:

```
📂 POS-Tagging
│── data/
│   └── daily547_3pos.txt             # Dataset in CoNLL format (word-POS pairs)
│── token_utils.py                    # Utility functions for tokenizing text and generating features
│── model.py                          # Logistic Regression model implementation and training function
│── grid_search.py                    # Hyperparameter tuning through grid search
│── predict.py                        # POS tagging prediction function
│── main.py                           # Main script for training, evaluating, and making predictions
│── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation (this file)

```


---

## Usage

To train the model and make predictions on sample sentences, follow these steps:

1. **Download the dataset**: Make sure you have the dataset `daily547_3pos.txt` in the `data` folder. This dataset should be in CoNLL format with word-POS pairs separated by tabs and sentences separated by empty lines.

2. **Run the main script**: To train the model, perform hyperparameter tuning, and make predictions, run the following command:

   ```bash
   python main.py
