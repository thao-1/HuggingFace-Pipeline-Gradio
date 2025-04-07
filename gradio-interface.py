import gradio as gr
from transformers import pipeline
import torch
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the classifier globally to avoid reloading
try:
    logger.info("Loading the model...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    classifier = None

def classify_text(text, labels):
    # Check if model loaded successfully
    if classifier is None:
        return {"Error": "Model failed to load. Please restart the application."}
    
    try:
        # Input validation
        if not text or not labels:
            return {"Error": "Please provide both text and labels"}
        
        # Process labels
        labels = [l.strip() for l in labels.split(",") if l.strip()]
        if not labels:
            return {"Error": "Please provide at least one valid label"}
        
        # Log inputs for debugging
        logger.info(f"Processing text: '{text}' with labels: {labels}")
        
        # Run classification
        result = classifier(text, candidate_labels=labels)
        
        # Log result for debugging
        logger.info(f"Classification result: {result}")
        
        # Explicitly convert NumPy values to Python floats
        return {label: float(score) for label, score in zip(result["labels"], result["scores"])}
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        return {"Error": f"An error occurred: {str(e)}"}

# Create the interface
interface = gr.Interface(
    fn=classify_text,
    inputs=[
        gr.Textbox(label="Input Text", placeholder="Enter text to classify..."),
        gr.Textbox(label="Candidate Labels (comma-separated)", placeholder="positive, negative, neutral")
    ],
    outputs=gr.Label(label="Classification Scores"),
    title="Zero-Shot Text Classifier",
    description="Classify any text into custom categories using zero-shot learning.",
    examples=[
        ["I love this movie, it's fantastic!", "positive, negative, neutral"],
        ["The weather is terrible today.", "positive, negative, neutral"],
        ["This product works well but is expensive.", "positive, negative, neutral"]
    ]
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()





