from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import torch
import numpy as np
from torch.nn.functional import softmax
from functools import wraps
import pdfplumber

from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    MobileBertTokenizer, MobileBertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

from config import Config

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.config.from_object(Config)

# -----------------------------
# Database setup
# -----------------------------
db = SQLAlchemy(app)

# -----------------------------
# Database Models
# -----------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class VisitorCount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    count = db.Column(db.Integer, default=0)

# -----------------------------
# Create tables & initialize visitor row
# -----------------------------
with app.app_context():
    db.create_all()
    if VisitorCount.query.first() is None:
        db.session.add(VisitorCount(count=0))
        db.session.commit()

# -----------------------------
# Helpers
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def current_visitors():
    vc = VisitorCount.query.first()
    return vc.count if vc else 0

def normalize_text(t: str, lower: bool = True) -> str:
    if not t:
        return ""
    t = t.replace("\x0c", " ")
    t = re.sub(r"http\S+", " <URL> ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if lower else t

def chunk_by_words(t: str, n: int = 180):
    words = t.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)] or [""]

def load_model(tokenizer_cls, model_cls, path):
    if not path or not os.path.exists(path):
        raise RuntimeError(f"Model path missing or invalid: {path}")
    tok = tokenizer_cls.from_pretrained(path)
    mdl = model_cls.from_pretrained(path).to(DEVICE).eval()
    return tok, mdl

def predict_batched(model, tokenizer, text: str, max_len: int, threshold: float):
    # Respect tokenizer casing if available
    lower = getattr(tokenizer, "do_lower_case", True)
    text = normalize_text(text, lower=lower)

    parts = chunk_by_words(text, n=180)
    enc = tokenizer(parts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits  # [batch, 2]
        probs = softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    avg_prob = float(np.mean(probs)) if len(probs) else 0.0
    return avg_prob >= threshold, avg_prob

def build_pred_obj(kind: str, flag: bool, prob: float):
    channels = {"Email": "📧 Email", "SMS": "📱 SMS", "News Article": "📰 News", "Social Media": "💬 Social Media"}
    labels = ("🚫 Spam", "✅ Not Spam") if kind in ("Email", "SMS") else ("🚫 Fake", "✅ Not Fake")
    return {
        "channel": channels.get(kind, kind),
        "flag": bool(flag),  # True => spam/fake
        "bad": labels[0],
        "ok": labels[1],
        "confidence": round(prob * 100, 1),  # storing as percentage

    }

ALLOWED_EXTENSIONS = {"pdf"}
def allowed_file(fn): 
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

MAX_LEN = 256
THRESHOLDS = {"Email": 0.90, "SMS": 0.85, "News Article": 0.80, "Social Media": 0.80}

# -----------------------------
# Load Models (fail fast with clear error)
# -----------------------------
try:
    email_tokenizer, email_model   = load_model(DistilBertTokenizer, DistilBertForSequenceClassification, app.config["EMAIL_MODEL_DIR"])
    sms_tokenizer, sms_model       = load_model(MobileBertTokenizer,  MobileBertForSequenceClassification,  app.config["SMS_MODEL_DIR"])
    news_tokenizer, news_model     = load_model(BertTokenizer,        BertForSequenceClassification,        app.config["NEWS_MODEL_DIR"])
    social_tokenizer, social_model = load_model(RobertaTokenizer,     RobertaForSequenceClassification,     app.config["SOCIAL_MODEL_DIR"])
except Exception as e:
    raise SystemExit(f"Model load failed: {e}")

# -----------------------------
# Visitor Tracking
# -----------------------------
@app.before_request
def track_visitors():
    if "visited" not in session:
        # Atomic-ish update for SQLite
        first = VisitorCount.query.first()
        if first:
            VisitorCount.query.filter_by(id=first.id).update({VisitorCount.count: VisitorCount.count + 1})
            db.session.commit()
        session["visited"] = True

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session["username"] = email
            return redirect(url_for("loader"))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

    return render_template("login.html", visitor_count=current_visitors())

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered. Please login.", "error")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(email=email, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html", visitor_count=current_visitors())

@app.route("/loader")
@login_required
def loader():
    return render_template("loader.html", visitor_count=current_visitors())

@app.route("/index", methods=["GET", "POST"])
@login_required
def index():
    extracted_text = None
    pdf_error = None
    prediction = None
    content_type = None

    if request.method == "POST":
        content_type = request.form.get("content_type")
        user_input = request.form.get("user_input", "").strip()
        pdf_file = request.files.get("pdf_file")

        # Handle PDF
        if not user_input and pdf_file and allowed_file(pdf_file.filename):
            try:
                extracted_text = ""
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        extracted_text += page.extract_text() or ""
                if extracted_text.strip():
                    user_input = extracted_text
                else:
                    pdf_error = "No selectable text found. If this is a scanned PDF, run OCR (e.g., 'ocrmypdf')."
            except Exception:
                pdf_error = "Failed to parse PDF."

        if not user_input:
            flash("Please provide text or upload a PDF.", "error")
            return render_template(
                "index.html",
                extracted_text=extracted_text,
                pdf_error=pdf_error,
                prediction=None,
                content_type=content_type,
                visitor_count=current_visitors()
            )

        # Prediction
        try:
            if content_type == "Email":
                flag, prob = predict_batched(email_model, email_tokenizer, user_input, MAX_LEN, THRESHOLDS["Email"])
                prediction = build_pred_obj("Email", flag, prob)
            elif content_type == "SMS":
                flag, prob = predict_batched(sms_model, sms_tokenizer, user_input, MAX_LEN, THRESHOLDS["SMS"])
                prediction = build_pred_obj("SMS", flag, prob)
            elif content_type == "News Article":
                flag, prob = predict_batched(news_model, news_tokenizer, user_input, MAX_LEN, THRESHOLDS["News Article"])
                prediction = build_pred_obj("News Article", flag, prob)
            elif content_type == "Social Media":
                flag, prob = predict_batched(social_model, social_tokenizer, user_input, MAX_LEN, THRESHOLDS["Social Media"])
                prediction = build_pred_obj("Social Media", flag, prob)
            else:
                flash("Please select a valid content type.", "error")
        except Exception as e:
            flash(f"Inference failed: {e}", "error")

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        pdf_error=pdf_error,
        prediction=prediction,
        content_type=content_type,
        visitor_count=current_visitors()
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
