# Tiny Sentiment API

A minimal sentiment analysis API that fine-tunes DistilBERT on IMDb reviews and serves predictions via FastAPI.

## Quickstart

### Local Development
```bash
# Clone and setup environment
git clone https://github.com/omerash1/tiny-sentiment-api.git
cd tiny-sentiment-api
conda create -n sentiment python=3.11
conda activate sentiment
pip install -r requirements.txt

# Train the model (optional - pre-trained weights available)
jupyter notebook notebooks/train_distilbert.ipynb

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t tiny-sentiment-api .
docker run -p 8000:8000 tiny-sentiment-api
```

## Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Predict Sentiment
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

**Response:**
```json
{
  "label": 1,
  "probability": 0.9542
}
```

- `label`: 0 (negative) or 1 (positive)
- `probability`: Confidence score for the predicted label

## Fine-tuning Details

- **Model**: `distilbert-base-uncased` 
- **Dataset**: IMDb movie reviews (10% subset for demo)
- **Training**: 2 epochs with default hyperparameters
- **Framework**: TensorFlow/Keras via Hugging Face Transformers

The model achieves ~90% accuracy on the IMDb test set.

## Model Weights

⚠️ **Important**: The `sentiment_model/` folder is excluded from git due to size. You need to train the model locally:

1. Run the training script: `python train_model.py` 
2. Or use the Jupyter notebook: `notebooks/train_distilbert.ipynb`

Training takes ~30 minutes and achieves 89%+ accuracy on IMDb reviews.
