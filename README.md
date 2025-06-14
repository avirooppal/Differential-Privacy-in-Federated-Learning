# Differential Privacy in Federated Learning with IoT Location Data

## Overview

This script demonstrates a simple implementation of **Federated Learning (FL)** combined with **Differential Privacy (DP)** using data from two IoT devices. Each device collects GPS location data (latitude and longitude), trains a local model, adds DP noise, and contributes to a global model update without sharing raw data.

## Scenario

Imagine two IoT devices (e.g., smart collars on pets) collecting GPS coordinates over time. Each device wants to contribute to a shared model (e.g., to detect common movement patterns) without exposing its exact location data.

## Code Breakdown

1. **Data Simulation**

   ```python
   device1_data = np.random.rand(100,2)
   device2_data = np.random.rand(100,2)
   ```

   * Simulates 100 samples of 2D location data (lat, long) for two devices.

2. **Local Training**

   ```python
   def local_train(data):
       return np.mean(data, axis=0)
   ```

   * Each device computes the **mean location** as its local model update.

3. **Differential Privacy (DP)**

   ```python
   def add_dp_noise(update, epsilon=1.0, sensitivity=1.0):
       scale = sensitivity / epsilon
       noise = np.random.laplace(0, scale, size=update.shape)
       return update + noise
   ```

   * **Laplace noise** is added to each device's local update to ensure privacy.
   * `epsilon` controls the privacy level (lower = more private).

4. **Federated Averaging**

   ```python
   global_model = (dp_update1 + dp_update2) / 2
   ```

   * The server averages the DP-noised updates from both devices to get the **global model**.

## Key Concepts

* **Federated Learning (FL)**: Devices train locally and share only model updates.
* **Differential Privacy (DP)**: Noise is added to updates so individual data points can't be inferred.
* This combination enables **privacy-preserving collaboration** between devices.

## Example Output

```bash
Update1 [0.51, 0.48]
Update2 [0.50, 0.52]
dp update1 [0.60, 0.41]
dp update2 [0.47, 0.55]
Global model: [0.53, 0.48]
```

Each device's contribution is masked by noise, but the global model still reflects useful aggregate information.

---


