# ⚽ FIFA Ranking Stability Predictor

An interactive **Streamlit web application** that analyzes **FIFA Men's Ranking data**, predicts ranking stability using a Machine Learning model, compares teams, and simulates ranking changes based on point adjustments.

The project uses:
- FIFA Ranking Dataset (2022-10-06)
- A trained ML model (`best_model.pkl`)
- Streamlit for UI
- Scikit-learn for model handling
- Pandas, NumPy for data processing

---

## 🚀 Features

### **1. Team Insights**
View:
- Current rank  
- Previous rank  
- Movement (Improved / Dropped / Stable)  
- ML Prediction: Stable / Changed  
- Probability score  
- Team's football association  

### ✔ **2. Compare Two Teams**
Compare any two teams side-by-side:
- Ranks  
- Points  
- ML predictions  

### ✔ **3. Ranking Trend Simulation**
Simulate ranking changes by adjusting points:
- Increase or decrease team points  
- See newly calculated rank  
- Visualize ranking change using bar graph  

---

## 📂 Project Structure

```
📦 FIFA Ranking Stability Predictor
│
├── README.md
├── app.py            
├── best_model.pkl       
├── fifa_ranking_2022-10-06.csv  
├── project.py                  
└── requirements.txt
    
```

---

## 🛠 Installation & Setup

### **1. Clone the Repository**
```
git clone https://github.com/your_username/fifa-ranking-predictor
cd fifa-ranking-predictor
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
```

### **3. Run the Streamlit App**
```
streamlit run app.py
```

---

## 📘 How the Model Works

The ML model predicts whether a team’s rank will be:

- `0` → Stable  
- `1` → Changed  

Using the features:
- Current Rank  
- Previous Rank  
- Current Points  
- Previous Points  

The model outputs:
- Prediction (Stable/Changed)
- Probability of change  

---

## 🤝 Contributing
Pull requests are welcome!  
Feel free to raise issues or suggest new features.

