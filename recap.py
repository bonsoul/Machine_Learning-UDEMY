import pandas as pd

#from sklearn import test_train_split

# line_plot.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sample data: dates + values
dates = pd.date_range("2025-01-01", periods=12, freq="M")
values = np.cumsum(np.random.randn(12)) + 10

df = pd.DataFrame({"date": dates, "value": values}).set_index("date")

plt.figure(figsize=(8,4))
plt.plot(df.index, df["value"], marker="o")
plt.title("Monthly values (example)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("line_plot.png")
plt.show()
