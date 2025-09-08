# Currency Conversion Matrix Auto-Updater
# Requirements: pip install requests pandas openpyxl

import requests
import pandas as pd

# Define currencies to include
currencies = ['USD', 'EUR', 'GBP', 'KES', 'ZAR', 'INR', 'CNY', 'NGN', 'UGX', 'TZS', 'RWF', 'JPY', 'CHF', 'AUD', 'CAD', 'BWP']

# Use ExchangeRate.host API (free, no key required)
base_currency = 'EUR'
url = f"https://api.exchangerate.host/latest?base={base_currency}"
response = requests.get(url)
data = response.json()
rates = data['rates']

# Filter to desired currencies
filtered_rates = {cur: rates[cur] for cur in currencies if cur in rates}
filtered_rates[base_currency] = 1.0  # Ensure base is included

# Build matrix
matrix = pd.DataFrame(index=filtered_rates.keys(), columns=filtered_rates.keys())

for sell in filtered_rates:
    for buy in filtered_rates:
        matrix.loc[sell, buy] = round(filtered_rates[buy] / filtered_rates[sell], 6)

# Save to files
matrix.to_excel("currency_conversion_matrix_live.xlsx")
matrix.to_csv("currency_conversion_matrix_live.csv")

print("Currency conversion matrix generated and saved.")