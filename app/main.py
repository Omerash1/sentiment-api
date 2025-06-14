import os
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Request/Response models
class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: int
    probability: float


# Global variables for model and tokenizer
model = None
tokenizer = None

# FastAPI app
app = FastAPI(
    title="Tiny Sentiment API",
    description="A minimal sentiment analysis API using fine-tuned DistilBERT",
    version="1.0.0"
)


@app.on_event("startup")
async def load_model():
    """Load the trained model and tokenizer at startup."""
    global model, tokenizer
    
    model_path = "app\sentiment_model"
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}. Please train the model first.")
    
    try:
        # Load the SavedModel
        model = tf.keras.models.load_model(model_path)
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        print(f"Model and tokenizer loaded successfully from {model_path}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


@app.get("/")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "message": "Tiny Sentiment API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """Predict sentiment for the given text."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            request.text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="tf"
        )
        
        # Make prediction
        outputs = model(inputs)
        
        # Handle different output formats
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, dict) and 'logits' in outputs:
            logits = outputs['logits']
        else:
            # For SavedModel format, outputs might be direct logits
            logits = outputs
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0]
        
        # Get predicted label and confidence
        predicted_label = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_label])
        
        return PredictionResponse(
            label=predicted_label,
            probability=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)