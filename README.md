Here is the updated `README.md` in English, incorporating your specific instructions for Kaggle authentication and environment setup.

---

# RL Project: Optimization of a Bid-Ask Strategy in Market Making

This project implements a Reinforcement Learning (RL) agent to optimize bid-ask quoting strategies within a simulated Limit Order Book (LOB). The agent interacts with an environment built from historical Bitcoin and ETH trading data to balance profitability and inventory risk.

## 1. Kaggle Authentication

The project requires access to Kaggle datasets. You must authenticate using one of the two methods below:

### Option A: Using `kaggle.json` (Recommended)

1. Log in to your [Kaggle](https://www.kaggle.com/) account.
2. Navigate to your `account settings` and click on **"Create Legacy API Key"**.
3. It will download the `kaggle.json` file.
4. Move the file to the following specific destination on your computer:
* **Windows**: `C:\Users\<Windows-username>\.kaggle\kaggle.json`
* **Linux/Mac**: `~/.kaggle/kaggle.json`



### Option B: Manual Credential Entry

In the notebook `market_making_analysis.ipynb`, you can manually set your environment variables by replacing the placeholders with your credentials from the JSON file:

```python
import os
# Replace with your actual username and key from the json file
os.environ['KAGGLE_USERNAME'] = "your_user_name"
os.environ['KAGGLE_KEY'] = "your_key"

```

*Note: If you encounter issues, feel free to contact me for my specific credentials.*

## 2. Environment Setup

To run this project, you must create a virtual environment and install the necessary dependencies:

1. **Create a virtual environment**:
```bash
python -m venv venv

```


2. **Activate the environment**:
* **Windows**: `venv\Scripts\activate`
* **Linux/Mac**: `source venv/bin/activate`


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```



## 3. Running the Analysis

Once the environment is ready and Kaggle is authenticated, you can execute the cells in the `market_making_analysis.ipynb` notebook.

The notebook show different graphics for analysis performances



The details and methodology of the project are fully described in **RL_Report.pdf**.
