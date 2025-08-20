# 🕵️ Fake & Spam Detection System  

A **Flask-based AI-powered detection system** that classifies text and PDF content into:  
- 📧 **Spam / Not Spam Emails**  
- 📱 **Spam / Not Spam SMS Messages**  
- 📰 **Fake / Real News Articles**  
- 💬 **Fake / Real Social Media Posts**  

Built using **state-of-the-art Transformer models** (DistilBERT, MobileBERT, BERT, RoBERTa), this project provides a **modern web interface**, secure login system, and **real-time classification with confidence scores**.  

---

## ✨ Key Features  

- 🔐 **Secure Login System** (with session handling)  
- 📂 **PDF Upload Support** (auto-detects content type: Email, SMS, News, or Social Media)  
- 📊 **Real-Time Predictions** with confidence score  
- 🖥️ **User-Friendly Web Interface** (Flask + HTML + CSS + Bootstrap)  
- ⚡ **Optimized Processing** using text normalization & chunking for large inputs  
- 📡 **Multi-Model Support**:  
  - DistilBERT → Spam Email Detection  
  - MobileBERT → SMS Spam Detection  
  - BERT → Fake News Detection  
  - RoBERTa → Fake Social Media Detection  

---

## 🖼️ Screenshots  

### 🔐 Login Page  
![Login Page]<img width="615" height="302" alt="image" src="https://github.com/user-attachments/assets/47efb129-d0f2-4b71-a7ab-c1b446ec431a" />


### 🕵️ Detection Dashboard  
![Detection Page]<img width="615" height="300" alt="image" src="https://github.com/user-attachments/assets/2157c245-003a-4efd-9f12-90a8eb97702e" />
 

---

## 🛠️ Tech Stack  

- **Backend:** Flask (Python)  
- **Frontend:** HTML, CSS, Bootstrap, Jinja2  
- **Machine Learning Models:**  
  - [DistilBERT](https://huggingface.co/distilbert-base-uncased) → Email Spam  
  - [MobileBERT](https://huggingface.co/google/mobilebert-uncased) → SMS Spam  
  - [BERT](https://huggingface.co/bert-base-uncased) → Fake News  
  - [RoBERTa](https://huggingface.co/roberta-base) → Social Media  
- **Libraries:** Torch, Transformers (Hugging Face), NumPy, Regex, pdfplumber  

---

## 📂 Project Structure  

├── app.py # Main Flask Application
├── models/ # Fine-tuned models (DistilBERT, MobileBERT, BERT, RoBERTa)
├── templates/ # HTML Templates (login.html, index.html, loader.html)
├── static/ # CSS, JS, Background Images
├── screenshots/ # Project Screenshots (login.png, detection.png)
├── requirements.txt # Python Dependencies
└── README.md # Documentation


---

## ⚡ Setup & Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/nagavallikareethu/Fake_News.git
cd Fake_News
```

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Flask App
python app.py

5️⃣ Open in Browser
http://127.0.0.1:5000


