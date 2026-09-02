# Trained AI Models

This directory contains the trained AI models for the EcoAgri Intelligence platform.

## Model Files

### Crop Classifier
- **File:** `crop_classifier/crop_classifier.keras`
- **Purpose:** Classify crop images into grades (Grade A, Grade B, Rejected)
- **Framework:** TensorFlow/Keras
- **Input:** Crop images (224x224 RGB)
- **Output:** Crop type, grade, confidence, freshness, quality scores

### Price Predictor
- **File:** `price_prediction/price_prediction.pkl`
- **Purpose:** Predict future crop prices based on historical data and market conditions
- **Framework:** Scikit-learn or similar
- **Input:** Historical prices, weather data, market conditions
- **Output:** Price forecasts, weekly predictions, monthly trends, best selling day

## Setup Instructions

### When Models Are Trained

1. **Place the crop classifier model:**
   ```
   trained_models/crop_classifier/crop_classifier.keras
   ```

2. **Place the price predictor model:**
   ```
   trained_models/price_prediction/price_prediction.pkl
   ```

3. **Enable model loading in the code:**
   - In `services/crop_classifier.py`, uncomment the TensorFlow model loading code
   - In `services/price_predictor.py`, uncomment the joblib model loading code
   - In `services/ai_service.py`, uncomment the actual prediction logic

### Current Status

The backend is currently set up to use **dummy predictions** when model files are not available. This ensures the application remains fully functional even without trained AI models.

When you place the trained model files in the correct locations and enable the loading code, the backend will automatically:
1. Load the models on startup
2. Use AI predictions instead of dummy data
3. Fall back to dummy predictions if model loading fails

## Model Training

### Dataset Structure

Training datasets should be organized in the `datasets/` directory:

```
datasets/
├── crop_grading/
│   ├── tomato/
│   │   ├── grade_a/
│   │   ├── grade_b/
│   │   └── rejected/
│   ├── banana/
│   └── onion/
└── mandi_prices/
```

### Training Scripts

Training scripts should be created in a separate `training/` directory (to be added later).

## Notes

- The backend checks for model files on startup
- If models are missing, it continues with dummy predictions
- No changes to the frontend are required when models are added
- The API endpoints remain the same regardless of whether AI or dummy predictions are used
