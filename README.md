# HuggingFace Zero-Shot Text Classifier with Gradio Frontend

This project implements a zero-shot text classification system using HuggingFace's transformers library with a Gradio web interface. It allows users to classify text into custom categories without needing to train a model.

## Features

- Zero-shot text classification using the BART-large-MNLI model
- Simple web interface built with Gradio
- Custom label support - you can define your own classification categories
- Real-time classification with confidence scores
- Example inputs provided for easy testing

## Requirements

- Python 3.9+
- PyTorch 2.0.1
- Transformers 4.30.2
- Gradio 4.44.1
- NumPy 1.24.3

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages using the requirements.txt file:
```bash
pip install -r requirements.txt
```

Alternatively, you can install the packages individually:
```bash
pip install torch==2.0.1 transformers==4.30.2 gradio numpy==1.24.3
```

## Usage

1. Run the application:
```bash
python gradio-interface.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7861)

3. Use the interface:
   - Enter the text you want to classify in the "Input Text" field
   - Enter your desired classification labels in the "Candidate Labels" field, separated by commas
   - Click "Submit" to see the classification results

## Example

Input Text:
```
I love this movie, it's fantastic!
```

Candidate Labels:
```
positive, negative, neutral
```

Expected Output:
```
{
    "positive": 0.95,
    "negative": 0.03,
    "neutral": 0.02
}
```

## How It Works

The application uses the BART-large-MNli model from Facebook/Meta AI, which is capable of zero-shot classification. This means it can classify text into categories it wasn't specifically trained on. The model:

1. Takes your input text and candidate labels
2. Evaluates how well the text matches each label
3. Returns confidence scores for each label

## Troubleshooting

If you encounter any issues:

1. Make sure you're using the exact package versions specified in the requirements
2. Ensure you have activated the virtual environment
3. Check that all dependencies are properly installed
4. If you see NumPy-related errors, make sure you're using NumPy 1.24.3
   

## Acknowledgments

- HuggingFace for the transformers library
- Facebook/Meta AI for the BART-large-MNLI model
- Gradio team for the web interface framework 
