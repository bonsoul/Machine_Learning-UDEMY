# Currency Conversion Matrix Auto-Updater
# Requirements: pip install requests pandas openpyxl numpy

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Define currencies to include
currencies = [
    'USD', 'EUR', 'GBP', 'KES', 'ZAR', 'INR', 'CNY', 'NGN',
    'UGX', 'TZS', 'RWF', 'JPY', 'CHF', 'AUD', 'CAD', 'BWP'
]

# Base currency for conversion
base_currency = 'EUR'
url = f"https://api.exchangerate.host/latest?base={base_currency}"

# Fetch data from API
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"❌ API request failed with status {response.status_code}")

data = response.json()

if "rates" not in data:
    raise Exception(f"❌ 'rates' key missing in API response: {data}")

rates = data["rates"]

# Filter to desired currencies
filtered_rates = {}
for cur in currencies:
    if cur in rates:
        filtered_rates[cur] = rates[cur]
    else:
        print(f"⚠️ Warning: {cur} not available in API response")

# Ensure base is included
filtered_rates[base_currency] = 1.0  

# Build conversion matrix using numpy broadcasting
curr_list = list(filtered_rates.keys())
values = np.array(list(filtered_rates.values()))
matrix_values = values[np.newaxis, :] / values[:, np.newaxis]

matrix = pd.DataFrame(matrix_values, index=curr_list, columns=curr_list).round(6)

# Save to files with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
excel_file = f"currency_conversion_matrix_{timestamp}.xlsx"
csv_file = f"currency_conversion_matrix_{timestamp}.csv"

matrix.to_excel(excel_file)
matrix.to_csv(csv_file)

print(f"✅ Currency conversion matrix generated and saved as:\n- {excel_file}\n- {csv_file}")
