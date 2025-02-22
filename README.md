# CAPTCHA AI Backend

A Flask-based backend service that provides CAPTCHA prediction and verification capabilities using a custom CNN model.

## Features

- Random CAPTCHA generation and serving
- CAPTCHA text prediction using a custom CNN model
- CAPTCHA verification endpoint
- Image preprocessing utilities
- Configuration management

## Tech Stack

- Python 3.8+
- Flask 2.3.3
- PyTorch 2.0.1
- Pillow 10.0.0
- Flask-CORS 4.0.0

## Project Structure

```
backend/
├── app/
│   ├── config/
│   │   └── config.py         # Configuration settings
│   ├── models/
│   │   └── inference.py      # CNN model and predictor
│   ├── routes/
│   │   └── captcha.py        # API endpoints
│   ├── utils/
│   │   └── image_processing.py # Image preprocessing utilities
│   └── __init__.py           # App initialization
├── tests/
│   └── test_endpoints.py     # API endpoint tests
└── run.py                    # Application entry point
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/captcha-ai-backend.git
cd captcha-ai-backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the model and data directories:
   - Place your trained model in `../captcha/models/run_20250209_212716/best_model.pt`
   - Add CAPTCHA images to `../captcha/data/raw/`

5. Run the application:
```bash
python run.py
```

The server will start at `http://localhost:5000`

## API Endpoints

### 1. Get Random CAPTCHA
```
GET /api/captcha/random
```
Returns a random CAPTCHA image and its AI prediction.

Response:
```json
{
    "success": true,
    "imagePath": "/api/captcha/image/abc123.png",
    "prediction": "ABC123",
    "confidence": 0.95,
    "confidencePerChar": [0.98, 0.97, 0.96, 0.94, 0.93, 0.92]
}
```

### 2. Verify CAPTCHA
```
POST /api/captcha/verify
```
Verifies user input against actual CAPTCHA text.

Request body:
```json
{
    "text": "ABC123",
    "imageFile": "/api/captcha/image/abc123.png"
}
```

Response:
```json
{
    "correct": true,
    "actualText": "ABC123"
}
```

### 3. Predict CAPTCHA
```
POST /api/captcha/predict
```
Predicts text from a CAPTCHA image.

Request: `multipart/form-data` with image file

Response:
```json
{
    "success": true,
    "prediction": "ABC123",
    "confidence": 0.95,
    "confidencePerChar": [0.98, 0.97, 0.96, 0.94, 0.93, 0.92]
}
```

## Model Architecture

The backend uses a custom CNN model with the following architecture:
- Input: Grayscale images (120x240 pixels)
- Feature extraction layers with residual blocks
- Global average pooling
- Separate classifier heads for each character
- Output: 5 characters, each with 62 classes (digits + lowercase + uppercase)

## Configuration

Key configuration settings in `config.py`:
- `MODEL_PATH`: Path to the trained model
- `IMAGE_SIZE`: Expected input image dimensions (120, 240)
- `SEQUENCE_LENGTH`: CAPTCHA text length (5)
- `NUM_CLASSES`: Number of possible characters (62)
- `ALLOWED_EXTENSIONS`: Allowed image file types

## Error Handling

The backend includes comprehensive error handling:
- Invalid file types
- Missing required fields
- Model loading errors
- Prediction errors
- File system errors

## Development

1. Enable debug mode in `run.py`:
```python
app.run(debug=True)
```

2. Run tests:
```bash
python -m pytest tests/
```

## Deployment Considerations

1. Security:
   - Replace `dev-key` with a secure `SECRET_KEY`
   - Implement rate limiting
   - Add request validation
   - Configure CORS properly

2. Performance:
   - Use production WSGI server (e.g., Gunicorn)
   - Implement caching
   - Consider model optimization

3. Monitoring:
   - Add logging
   - Implement health checks
   - Monitor model performance

## License

MIT License - Feel free to use this code for your own projects.
