Federated Learning with Differential Privacy for IoT Location Data
Overview
This project demonstrates Federated Learning (FL) combined with Differential Privacy (DP) on simulated GPS data from IoT devices. It includes two scripts:

main.py: Simulates 2 IoT devices sharing noisy local updates.

main2.py: Measures how the DP error changes as the number of devices increases.

main.py — Basic FL + DP with Two Devices
Scenario
Two IoT devices (e.g., GPS trackers) collect 2D location data (latitude, longitude). They want to collaboratively build a global model (e.g., average location), without exposing their private data.

How It Works
Data Simulation: Each device generates 100 samples of 2D location data.

Local Training: Devices compute the mean of their local data.

Differential Privacy: Laplace noise is added to each local update.

Global Aggregation: The server averages the noisy updates to produce a global model.

Code Summary
python
Copy
Edit
update = np.mean(device_data, axis=0)
dp_update = update + Laplace_noise
global_model = average(dp_update1, dp_update2)
Key Points
Protects privacy using Laplace Mechanism.

Enables decentralized collaboration without raw data sharing.

main2.py — Effect of Number of Devices on DP Error
Goal
Quantify how increasing the number of devices reduces the error introduced by DP noise.

What It Does
Runs FL+DP with varying numbers of devices: [2, 5, 10, 20, 50, 100].

Each device trains locally and adds DP noise.

Compares the DP global model with the true (non-noisy) global model using Euclidean distance.

Plots the DP error vs number of devices.

Key Insight
More devices → Better averaging → Lower overall noise effect → Smaller error.

Plot Example
javascript
Copy
Edit
Y-axis: Error (L2 norm between noisy and true global means)
X-axis: Number of devices
The plot shows that as the number of devices increases, the impact of noise diminishes.

Summary
This project shows how Federated Learning and Differential Privacy can be combined to enable privacy-preserving collaboration between IoT devices, especially for sensitive data like location.

Privacy Tradeoff
ε (epsilon) controls privacy: smaller ε = more privacy, but more noise.

Averaging across more devices reduces the distortion from noise.
