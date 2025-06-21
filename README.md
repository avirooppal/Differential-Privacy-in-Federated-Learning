# Federated Learning with Differential Privacy for IoT Location Data

## Overview

This project demonstrates **Federated Learning (FL)** combined with **Differential Privacy (DP)** on simulated GPS data from IoT devices. It includes two scripts:

- `main.py`: Simulates 2 IoT devices sharing noisy local updates.
- `main2.py`: Measures how the DP error changes as the number of devices increases.

---

## `main.py` — Basic FL + DP with Two Devices

### Scenario

Two IoT devices (e.g., GPS trackers) collect 2D location data (latitude, longitude). They want to collaboratively build a global model (e.g., average location), without exposing their private data.

### How It Works

- **Data Simulation**: Each device generates 100 samples of 2D location data.
- **Local Training**: Devices compute the mean of their local data.
- **Differential Privacy**: Laplace noise is added to each local update.
- **Global Aggregation**: The server averages the noisy updates to produce a global model.

### Code Summary

```python
update = np.mean(device_data, axis=0)
dp_update = update + Laplace_noise
global_model = (dp_update1 + dp_update2) / 2
```

### Key Points

- Protects privacy using the Laplace Mechanism.
- Enables decentralized collaboration without sharing raw data.

---

## `main2.py` — Effect of Number of Devices on DP Error

### Goal

Quantify how increasing the number of devices reduces the error introduced by DP noise.

### What It Does

- Runs FL+DP with varying numbers of devices: `[2, 5, 10, 20, 50, 100]`.
- Each device trains locally and adds DP noise to its update.
- Computes the true global mean (without noise) and the DP global mean (with noise).
- Measures the error as the Euclidean distance (L2 norm) between the two global models.
- Plots error versus number of devices.

### Key Insight

More devices → Better averaging → Lower overall noise effect → Smaller error.

### Plot Example

- **Y-axis**: Error (L2 norm between DP and true global means)  
- **X-axis**: Number of devices  

The plot shows that as the number of devices increases, the impact of noise diminishes due to averaging.

---

## Summary

This project shows how **Federated Learning** and **Differential Privacy** can be combined to enable **privacy-preserving collaboration** between IoT devices, especially when handling sensitive data such as location.

### Privacy Tradeoff

- **Epsilon (ε)** controls the privacy level:
  - Smaller ε → more noise → stronger privacy
  - Larger ε → less noise → higher accuracy
- Averaging across more devices reduces the effect of individual noise, improving model quality.



## `Federated Learning with Differential Privacy.ipynb`
# Overview
Dataset: MNIST (handwritten digits)

Clients: 10 (each with a non-IID data split)

Model: Simple fully connected network (784 → 10)

Training: 20 communication rounds, 1 local epoch per client per round

Privacy: Differential Privacy via gradient clipping and Gaussian noise addition

# Training Configuration

| Parameter         | Value                   |
| ----------------- | ----------------------- |
| Batch Size        | 32                      |
| Learning Rate     | 0.1                     |
| Gradient Clipping | 1.0 (L2 norm)           |
| Noise Multipliers | 0.0, 0.1, 0.5, 1.0, 2.0 |

# Final Accuracy by Noise Level

| Noise Multiplier | Final Test Accuracy |
| ---------------- | ------------------- |
| 0.0              | 0.7592              |
| 0.1              | 0.6720              |
| 0.5              | 0.4627              |
| 1.0              | 0.5084              |
| 2.0              | 0.4518              |

As expected, higher noise levels (stronger privacy) generally reduce model accuracy.

# Privacy vs. Utility Trade-off
![image](https://github.com/user-attachments/assets/529c6656-63f1-4de3-9b4a-31768ebf31d5)




