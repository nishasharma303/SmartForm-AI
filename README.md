# 🏗 SmartForm AI  
### AI-Driven Formwork Kitting & BoQ Optimization

SmartForm AI is a decision-support web application that optimizes formwork kitting in high-rise construction projects using unsupervised machine learning and constraint-based planning logic.

---

## 🚀 Problem Overview

Formwork contributes a significant portion of construction project costs.  
Manual BoQ estimation often leads to:

- Excess inventory  
- High carrying costs  
- Inefficient reuse planning  
- Static estimation without repetition analytics  

SmartForm AI introduces automated repetition detection and optimized kit planning to improve material efficiency.

---

## 🧠 How It Works

### 1️⃣ Data Input

The system accepts floor-wise structural parameters:

- Slab area  
- Beam length  
- Column count  
- Wall area  
- Cycle duration  

Users can:

- Upload their own CSV dataset  
- Or use the built-in sample 30-floor tower dataset  

---

### 2️⃣ Repetition Detection (AI Layer)

The application uses **KMeans Clustering (Unsupervised Learning)** to:

- Automatically group structurally similar floors  
- Detect repetition patterns  
- Identify reuse potential  

---

### 3️⃣ Kit Optimization Logic


This models:

- Kit reuse  
- Resource constraints  
- Parallel casting limitations  

---

### 4️⃣ Cost Evaluation

Total Cost = Procurement Cost + Carrying Cost  

Where:

- Procurement Cost = Kits × Kit Cost  
- Carrying Cost = % of Procurement Cost  

The dashboard displays:

- Kits Before (Manual assumption)  
- Kits After (Optimized planning)  
- Total Cost Before  
- Total Cost After  
- Estimated Savings  

---

## 📊 Features

- Executive KPI dashboard  
- AI-based repetition clustering  
- Cost comparison visualization  
- Adjustable scenario parameters  
- CSV upload support  
- Downloadable clustered floor plan  

---

## 🛠 Tech Stack

**Frontend**
- Streamlit  

**Backend**
- Python  

**Data Processing**
- Pandas  
- NumPy  

**Machine Learning**
- Scikit-learn (KMeans Clustering)  

**Deployment**
- Streamlit Community Cloud  
- GitHub  


