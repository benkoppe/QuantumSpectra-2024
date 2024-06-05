import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from pathlib import Path

parent_dir = Path(__file__).resolve().parent

data = pd.read_csv(
    f"{parent_dir}/sample_experiment.dat", sep="\s+", header=None, skiprows=2
)

x, y = np.array(data[1]), np.array(data[0])

range_filter = np.where(x < 10_000)
x, y = x[range_filter], y[range_filter]

# y_smooth = savgol_filter(y, window_length=400, polyorder=1)
y_smooth = UnivariateSpline(x, y, s=0.154e6, k=5)(x)


if __name__ == "__main__":
    plt.plot(x, y, label="Raw data")
    plt.plot(x, y_smooth, color="red", label="Smoothed data")
    plt.legend()
    plt.savefig(f"{parent_dir}/sample_experiment.png")
