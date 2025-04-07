from transformers import pipeline

# Load the zero-shot classification pipeline with a better model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Test input
text = "This new iPhone has an amazing camera but is too expensive."

# Candidate labels
labels = ["technology", "sports", "business", "politics", "entertainment"]

# Run classification
result = classifier(text, candidate_labels=labels)

# Display result
print("Input:", text)
print("Predicted Label:", result["labels"][0])
print("Scores:", list(zip(result["labels"], result["scores"])))

