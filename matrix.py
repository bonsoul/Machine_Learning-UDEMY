# Currency Conversion Matrix Auto-Updater
# Requirements:
#   pip install requests pandas openpyxl numpy

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# 1. Define currencies to include
# -----------------------------
currencies = [
    'USD', 'EUR', 'GBP', 'KES', 'ZAR', 'INR', 'CNY', 'NGN',
    'UGX', 'TZS', 'RWF', 'JPY', 'CHF', 'AUD', 'CAD', 'BWP'
]

# -----------------------------
# 2. Base currency for conversion
# -----------------------------
base_currency = 'EUR'
API_KEY = "8370f5b504580a3c9e76b8de92c6cea4"
url = f"https://api.exchangerate.host/latest?base={base_currency}&access_key={API_KEY}"

# -----------------------------
# 3. Fetch data from API
# -----------------------------
try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
except requests.RequestException as e:
    raise SystemExit(f"❌ API request failed: {e}")

data = response.json()

if "rates" not in data:
    raise KeyError(f"❌ 'rates' key missing in API response: {data}")

rates = data["rates"]

# -----------------------------
# 4. Filter to desired currencies
# -----------------------------
filtered_rates = {}
for cur in currencies:
    if cur in rates:
        filtered_rates[cur] = rates[cur]
    else:
        print(f"⚠️ Warning: {cur} not available in API response")

# Ensure base currency is always included
filtered_rates[base_currency] = 1.0  

# -----------------------------
# 5. Build conversion matrix
# -----------------------------
curr_list = list(filtered_rates.keys())
values = np.array(list(filtered_rates.values()), dtype=float)

matrix_values = values[np.newaxis, :] / values[:, np.newaxis]
matrix = pd.DataFrame(matrix_values, index=curr_list, columns=curr_list).round(6)

# -----------------------------
# 6. Save to files with timestamp
# -----------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_file = f"currency_conversion_matrix_{timestamp}.xlsx"
csv_file = f"currency_conversion_matrix_{timestamp}.csv"

matrix.to_excel(excel_file)
matrix.to_csv(csv_file)

print(f"✅ Currency conversion matrix generated and saved as:\n- {excel_file}\n- {csv_file}")
