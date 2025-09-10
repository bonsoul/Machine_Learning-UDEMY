import requests
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# 1. Full currency list (your desired set)
# -----------------------------
currencies = [
    'USD', 'EUR', 'GBP', 'KES', 'ZAR', 'INR', 'CNY', 'NGN',
    'UGX', 'TZS', 'RWF', 'JPY', 'CHF', 'AUD', 'CAD', 'BWP'
]

BASE = "EUR"

# -----------------------------
# 2. Get supported currencies from Frankfurter
# -----------------------------
resp = requests.get("https://api.frankfurter.app/currencies", timeout=10)
resp.raise_for_status()
supported = set(resp.json().keys())

# Filter your list to only supported ones
usable_currencies = [cur for cur in currencies if cur in supported]
if BASE not in usable_currencies:
    usable_currencies.append(BASE)

print(f"✅ Supported currencies used: {usable_currencies}")
skipped = [cur for cur in currencies if cur not in supported]
if skipped:
    print(f"⚠ Skipped unsupported currencies: {skipped}")

# -----------------------------
# 3. Fetch latest rates
# -----------------------------
symbols = ",".join([cur for cur in usable_currencies if cur != BASE])
url = f"https://api.frankfurter.app/latest?from={BASE}&to={symbols}"

resp = requests.get(url, timeout=10)
resp.raise_for_status()
data = resp.json()

rates = data["rates"]
rates[BASE] = 1.0

# -----------------------------
# 4. Build conversion matrix
# -----------------------------
curr_list = list(rates.keys())
values = np.array([rates[cur] for cur in curr_list], dtype=float)

matrix_values = values[np.newaxis, :] / values[:, np.newaxis]
matrix = pd.DataFrame(matrix_values, index=curr_list, columns=curr_list).round(6)

# -----------------------------
# 5. Save to Excel & CSV
# -----------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
matrix.to_excel(f"currency_matrix_{timestamp}.xlsx")
matrix.to_csv(f"currency_matrix_{timestamp}.csv")

print("✅ Conversion matrix saved successfully.")
