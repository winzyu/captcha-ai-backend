# run.py
from flask import Flask
from flask_cors import CORS
from app.routes.captcha import captcha_bp
from app.config.config import Config

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Load and verify configuration
    app.config.from_object(Config)
    Config.init_app()
    
    # Register blueprints
    app.register_blueprint(captcha_bp)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
