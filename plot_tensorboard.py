import pandas as pd
import matplotlib.pyplot as plt

# Paths to your CSV files
train_csv_path = r"C:\UCLA\SungLab\DiffusionMBIR\logs\tensorboard\exp1_b64_0629\training_loss.csv"
eval_csv_path = r"C:\UCLA\SungLab\DiffusionMBIR\logs\tensorboard\exp1_b64_0629\eval_loss.csv"

# Load the CSVs
train_df = pd.read_csv(train_csv_path)
eval_df = pd.read_csv(eval_csv_path)

# Fix the step indices assuming:
# - training_loss logged every 10 steps
# - eval_loss logged every 50 steps
train_df['FixedStep'] = range(10, 10 * (len(train_df) + 1), 10)
eval_df['FixedStep'] = range(50, 50 * (len(eval_df) + 1), 50)

# Plot both
plt.figure(figsize=(10, 5))
plt.plot(train_df['FixedStep'], train_df['Value'], label='Training Loss', color='blue')
plt.plot(eval_df['FixedStep'], eval_df['Value'], label='Eval Loss', color='red')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss vs. Step')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()