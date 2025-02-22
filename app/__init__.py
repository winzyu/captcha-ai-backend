# app/__init__.py
from flask import Flask
from flask_cors import CORS
from app.routes.captcha import captcha_bp
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    app.config.from_object(Config)
    app.register_blueprint(captcha_bp)
    
    return app
