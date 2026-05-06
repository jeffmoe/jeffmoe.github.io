---
title: PandasAI with Local LLM
parent: Machine Learning and Artificial Intelligence
nav_order: 3
---

### Project Overview
This project compares basic data filtering tasks between **Pandas** and **PandasAI** using a customer bank churn dataset from Kaggle. The goal is to evaluate the effectiveness and simplicity of AI-assisted data querying versus traditional Pandas filtering methods.

### Key Comparisons
- Filtering by **one column**
- Filtering by **two columns**
- Filtering by **three columns**

### Dataset
**Source:** [Customer Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling/data) (Kaggle)

### Environment

| Component | Version / Specification |
|-----------|--------------------------|
| Python | 3.10.0 |
| PandasAI | 3.0.0 |
| OS | WSL2 Debian instance |
| LLM | Local instance (LM Studio) running on native Windows (GPU) |
| Virtual Environment | Used to avoid dependency conflicts |

> **GitHub Repository:** [https://github.com/jeffmoe/CST_635_Week5.git](https://github.com/jeffmoe/CST_635_Week5.git)

### LLM Local Setup (LM Studio)
#### Steps to Configure LM Studio
1. Go to the **Developer** tab
2. Click the **gear icon** (top left) for server settings
3. Ensure:
   - **Require authentication** is OFF
   - **Serve on local network** is ON
4. Toggle the **server on** (top left)
5. **Load a model** (e.g., `gemma-4-e4b`)
6. Copy the **URL** provided by LM Studio (used for PandasAI setup)

### PandasAI Integration
#### Installation
```bash
pip install pandasai pandasai-litellm
```
### Code
```python

import pandas as pd
from pandasai import SmartDataframe
from pandasai_litellm import LiteLLM
from litellm import completion
import os

os.environ["API_BASE_URL"] = "http://10.0.0.78:1234/v1"
llm = LiteLLM(
    model="lm_studio/gemma-4-e4b",
    api_base=os.getenv("API_BASE_URL"),
    api_key="fake-key",
    temperature=0.2,
    max_tokens=2000
)
df = pd.read_csv('Churn_Modelling.csv')
sdf = SmartDataframe(df, config={
    "llm": llm,
    "enable_cache": False
})
response = sdf.chat("How many customers are in Germany?")
print(response)
2509
df[df['Geography'] == 'Germany'].value_counts().sum()
2509
response = sdf.chat("How many male customers are in Spain?")
print(response)
1388
df[(df['Geography'] == 'Spain') & (df['Gender'] == 'Male')].value_counts().sum()
1388
response = sdf.chat("How many female customers churned in France?")
print(response)
460
df[(df['Geography'] == 'France') & (df['Gender'] == 'Female') & (df['Exited'] == 1)].value_counts().sum()
460
```
### Outcomes
- Hands on experience with AI machine learning tools
- Experience with LM Studio
- Greatly reduced EDA time and code readability
