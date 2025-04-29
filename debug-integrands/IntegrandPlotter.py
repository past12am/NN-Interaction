import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing the CSV files
folder_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/debug-integrands/"  # Change this to your folder

# Regex pattern to extract X, Z, and eps from filename
pattern = re.compile(r"_X=(.*?)_Z=(.*?)_eps=(.*?)\.csv")

# Iterate over all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        match = pattern.search(filename)
        if match:
            X, Z, eps = match.groups()
            filepath = os.path.join(folder_path, filename)

            # Load CSV into DataFrame
            df = pd.read_csv(filepath)

            # Plot
            plt.figure()
            plt.plot(df['x4'], df['kernel_real'], label='Real Part')
            plt.plot(df['x4'], df['kernel_imag'], label='Imaginary Part')
            plt.title(f"Kernel Plot - X={X}, Z={Z}, ε={eps}")
            plt.xlabel("x")
            plt.ylabel("Kernel Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
