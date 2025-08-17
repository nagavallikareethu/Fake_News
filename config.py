import os

class Config:
    # Security & Flask
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32))
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///users.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Uploads
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

    # Session cookies
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("FLASK_ENV") == "production"

    # Model directories (set env vars or use absolute paths)
    EMAIL_MODEL_DIR  = os.getenv("EMAIL_MODEL_DIR",  r"D:\project\models\fine_tuned_distilbert_email\fine_tuned_distilbert_email")
    SMS_MODEL_DIR    = os.getenv("SMS_MODEL_DIR",    r"D:\project\models\fine_tuned_mobilebert_sms\fine_tuned_mobilebert_sms")
    NEWS_MODEL_DIR   = os.getenv("NEWS_MODEL_DIR",   r"D:\project\models\fine_tuned_bert_news\fine_tuned_bert_news")
    SOCIAL_MODEL_DIR = os.getenv("SOCIAL_MODEL_DIR", r"D:\project\models\roberta_social_model\fine_tuned_roberta_social")
