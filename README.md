# 🛠️ AI-Powered Service Center Assistant

## 📌 Project Overview

AI-Powered Service Center Assistant is a complete Machine Learning and AI-based web application designed to help service centers manage customer service requests efficiently.

This project helps service centers analyze service data, classify customer complaints, predict repair completion time, analyze customer feedback sentiment, and provide automated customer support using an AI chatbot.

The application is built using Python, Machine Learning, Natural Language Processing, Streamlit, FAISS, Sentence Transformers, and Hugging Face tools.

---

## 🎯 Project Objective

The main objective of this project is to automate service center operations using AI and Machine Learning.

The system can:

- Analyze service center data
- Classify customer complaints automatically
- Predict repair completion time
- Analyze customer feedback sentiment
- Perform complaint text analysis
- Provide AI-based customer support
- Track repair status using Customer ID
- Display everything in an interactive Streamlit dashboard

---

## 🚀 Key Features

### 1. Service Data Analysis

The dashboard provides complete Exploratory Data Analysis.

Features:

- Missing value analysis
- Duplicate record check
- Data type analysis
- Summary statistics
- Complaint category analysis
- Brand-wise complaint analysis
- Warranty status analysis
- Repair cost distribution
- Repair time distribution
- Customer rating analysis
- Monthly service request trend
- Technician performance analysis
- Priority-wise request analysis
- Correlation heatmap
- Interactive Plotly visualizations
- CSV upload and download option
- Customer record search

---

### 2. Service Ticket Classification

The ticket classification module uses NLP and Machine Learning to classify customer complaints.

Input:

```text
Customer complaint

Output:

Complaint Type

Complaint categories:

Battery Issue
Display Issue
Software Issue
Charging Issue
Camera Issue
Speaker Issue
Network Issue
Motherboard Issue

Models used:

Logistic Regression
Linear SVM
Naive Bayes
Random Forest Classifier

Evaluation metrics:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix

The best model is saved using Joblib.

3. Complaint Analysis

This module performs NLP-based complaint analysis.

Features:

Word cloud
Top words
Bi-grams
Tri-grams
TF-IDF keyword extraction
Complaint frequency analysis
Complaint trend analysis
Topic modeling using LDA
4. Repair Time Prediction

The repair prediction module predicts estimated repair completion time.

Target variable:

Repair_Time

Features used:

Complaint Type
Brand
Priority
Warranty Status
Parts Replaced
Repair Cost

Models used:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor

Evaluation metrics:

MAE
RMSE
R² Score

The best model is saved and used for real-time prediction in the Streamlit dashboard.

5. Customer Sentiment Analysis

This module analyzes customer feedback and classifies it as:

Positive
Neutral
Negative

Sentiment methods used:

TextBlob
NLTK VADER
Hugging Face Transformers

The module also compares the output of different sentiment analysis methods.

6. Advanced AI Chat Assistant

The AI chatbot helps customers ask service-related questions.

The chatbot can answer questions about:

Service center timings
Warranty policy
Out-of-warranty charges
Diagnostic charges
Battery issues
Screen replacement
Software reset
Charging problems
Network issues
Speaker issues
Motherboard repair time
Repair tracking using Customer ID

Advanced chatbot features:

RAG-based architecture
FAISS vector search
Sentence Transformer embeddings
Intent detection
Customer ID repair tracking
Conversation history
Confidence score
Retrieved knowledge base context
Fallback answers for stable performance
🧰 Technologies Used
Programming Language
Python
Data Processing
Pandas
NumPy
Scipy
Visualization
Matplotlib
Plotly
WordCloud
Machine Learning
Scikit-Learn
Joblib
NLP and Sentiment Analysis
NLTK
TextBlob
Regex
Hugging Face Transformers
AI Chatbot / RAG
LangChain
FAISS
Sentence Transformers
Hugging Face
PyTorch
Web Application
Streamlit
📁 Folder Structure
AI_Powered_Service_Center_Assistant/
│
├── data/
│   ├── service_center_data.csv
│   └── clean_service_center_data.csv
│
├── models/
│   ├── ticket_classifier.pkl
│   ├── repair_prediction.pkl
│   ├── tfidf.pkl
│   ├── classification_report.txt
│   └── model_metrics.csv
│
├── chatbot/
│   ├── __init__.py
│   ├── knowledge_base.txt
│   ├── rag.py
│   └── chatbot.py
│
├── pages/
│   ├── 1_Data_Analysis.py
│   ├── 2_Ticket_Classification.py
│   ├── 3_Complaint_Analysis.py
│   ├── 4_Repair_Time_Prediction.py
│   ├── 5_Sentiment_Analysis.py
│   ├── 6_AI_Assistant.py
│   └── 7_About_Project.py
│
├── utils/
│
├── logs/
│   └── training.log
│
├── app.py
├── data.py
├── train_models.py
├── requirements.txt
└── README.md
📊 Dataset Description

The project uses a realistic generated service center dataset containing 2000 records.

Important columns:

Column Name	Description
Customer_ID	Unique customer ID
Customer_Name	Name of the customer
Product	Product type
Product_Category	Category of product
Brand	Product brand
Purchase_Date	Date of purchase
Service_Request_Date	Date of service request
Warranty_Status	Warranty information
Complaint	Customer complaint text
Complaint_Type	Type of complaint
Priority	Ticket priority
Technician	Assigned technician
Repair_Time	Repair duration in days
Parts_Replaced	Replaced part
Repair_Cost	Repair cost
Status	Ticket status
Customer_Rating	Customer rating
Feedback	Customer feedback
Sentiment	Sentiment label
⚙️ Installation and Setup
Step 1: Clone or download the project
cd AI_Powered_Service_Center_Assistant
Step 2: Install required packages
python -m pip install -r requirements.txt
Step 3: Generate dataset
python data.py

This will create:

data/service_center_data.csv
data/clean_service_center_data.csv
Step 4: Train machine learning models
python train_models.py

This will create:

models/ticket_classifier.pkl
models/repair_prediction.pkl
models/tfidf.pkl
models/classification_report.txt
models/model_metrics.csv
Step 5: Run Streamlit application
python -m streamlit run app.py

or

streamlit run app.py
🧠 Machine Learning Workflow
Ticket Classification Workflow
Customer Complaint
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Classification Model
        ↓
Complaint Type Prediction
Repair Time Prediction Workflow
Service Details
        ↓
Feature Encoding
        ↓
Feature Scaling
        ↓
Regression Model
        ↓
Repair Time Prediction
Sentiment Analysis Workflow
Customer Feedback
        ↓
TextBlob / NLTK VADER / Hugging Face
        ↓
Sentiment Classification
        ↓
Positive / Neutral / Negative
AI Chatbot Workflow
User Question
        ↓
Intent Detection
        ↓
Knowledge Base Retrieval
        ↓
Sentence Transformer Embeddings
        ↓
FAISS Similarity Search
        ↓
AI Response
📈 Model Performance
Ticket Classification

The ticket classification model compares multiple ML algorithms:

Logistic Regression
Linear SVM
Naive Bayes
Random Forest

The best model is selected based on F1 Score.

Repair Time Prediction

The repair prediction model compares:

Linear Regression
Decision Tree
Random Forest
Gradient Boosting

The best model is selected based on R² Score and error metrics.

🖥️ Dashboard Pages

The Streamlit dashboard contains the following pages:

Home Dashboard
Service Data Analysis
Ticket Classification
Complaint Analysis
Repair Time Prediction
Sentiment Analysis
AI Assistant
About Project
🤖 AI Chatbot Demo Questions

You can test the chatbot using these questions:

What are your service center timings?
Is physical damage covered under warranty?
What is the battery replacement cost?
How much does screen replacement cost?
How long will motherboard replacement take?
What is the diagnostic charge?
Is software reset free under warranty?
My phone is not charging. What could be the reason?
WiFi is not connecting. What should I do?
Track repair status for CUST00001
📸 Screenshots

Add your project screenshots here.

Example:

### Home Dashboard
![Home Dashboard](screenshots/home_dashboard.png)

### Ticket Classification
![Ticket Classification](screenshots/ticket_classification.png)

### AI Assistant
![AI Assistant](screenshots/ai_assistant.png)
✅ Final Output

This project successfully demonstrates:

Data analysis
Machine learning classification
Machine learning regression
NLP complaint analysis
Customer sentiment analysis
RAG-based AI chatbot
Interactive dashboard development
Real-time prediction system
Model serialization
Customer repair tracking
🔮 Future Improvements

Future enhancements can include:

Use real service center data
Add login system for admin and customers
Add MySQL or SQLite database
Add SMS/email notification system
Add live repair tracking portal
Add advanced LLM API chatbot
Add multilingual chatbot support
Add customer satisfaction prediction
Deploy the application on Streamlit Cloud or Render
Add role-based access control
🧪 How to Test

Run the app:

python -m streamlit run app.py

Test ticket classification:

Battery drains very fast and phone heats while charging

Expected output:

Battery Issue

Test repair prediction:

Complaint Type: Battery Issue
Brand: Samsung
Priority: High
Warranty Status: Out of Warranty
Parts Replaced: Battery
Repair Cost: 2500

Expected output:

Estimated repair time around 5 days

Test sentiment analysis:

It is not good!

Expected output:

Negative

Test chatbot:

Track repair status for CUST00001

Expected output:

Customer repair details

👨‍💻 Author

Ranish A

B.Tech Artificial Intelligence and Data Science

github link = https://github.com/rranish887-debug/AI_Powered_Service_Center_Assistant.git

📜 License

This project is created for educational, internship, hackathon, and portfolio purposes.