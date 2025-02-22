# app/models/inference.py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CaptchaCNN(nn.Module):
    def __init__(self, num_chars=5, num_classes_per_char=62):
        super(CaptchaCNN, self).__init__()
        
        # Initial layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Deep feature extraction with residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Calculate feature dimensions
        with torch.no_grad():
            x = torch.zeros(1, 1, 50, 100)
            x = self.conv1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avg_pool(x)
            self.feature_size = x.view(1, -1).size(1)
        
        # Separate classifier heads for each character
        self.char_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes_per_char)
            ) for _ in range(num_chars)
        ])
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Convert to grayscale if input is RGB
        if x.shape[1] == 3:
            x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        
        # Feature extraction
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Get predictions for each character position
        return [classifier(x) for classifier in self.char_classifiers]

class CaptchaPredictor:
    def __init__(self, model_path: Path):
        # Force CPU usage to avoid CUDA issues
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model(model_path)
        self.char_to_idx = self._create_char_mapping()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
    
    def _create_char_mapping(self) -> Dict[str, int]:
        """Create character to index mapping"""
        return {
            **{str(i): i for i in range(10)},  # digits 0-9
            **{chr(i): i-87 for i in range(97, 123)},  # lowercase a-z
            **{chr(i): i-29 for i in range(65, 91)}    # uppercase A-Z
        }
    
    def _load_model(self, model_path: Path) -> nn.Module:
        """Load the PyTorch model"""
        try:
            print(f"Loading model from {model_path}")
            # Create model instance
            model = CaptchaCNN(num_chars=5, num_classes_per_char=62)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            print("Model loaded successfully")
            
            # Set to evaluation mode and move to correct device
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Predict CAPTCHA text from image tensor
        Returns prediction details including confidence scores
        """
        try:
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            
            # Get predictions and probabilities
            predictions = []
            confidences = []
            
            for output in outputs:
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, dim=1)
                predictions.append(self.idx_to_char[pred.item()])
                confidences.append(confidence.item())
            
            result = {
                'text': ''.join(predictions),
                'confidence_per_char': confidences,
                'average_confidence': sum(confidences) / len(confidences)
            }
            print(f"Prediction successful: {result}")
            return result
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
