import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  

def generate_device_data(n_samples=100):
    return np.random.rand(n_samples, 2)

def local_train(data):
    return np.mean(data, axis=0)

def add_dp_noise(update, epsilon=1.0, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=update.shape)
    return update + noise

device_counts = [2, 5, 10, 20, 50, 100]
errors = []

for num_devices in device_counts:
    updates = []
    dp_updates = []

    for _ in range(num_devices):
        data = generate_device_data()
        local_update = local_train(data)
        dp_update = add_dp_noise(local_update, epsilon=1.0)
        updates.append(local_update)
        dp_updates.append(dp_update)

    true_global = np.mean(updates, axis=0)
    dp_global = np.mean(dp_updates, axis=0)

    error = np.linalg.norm(dp_global - true_global)  
    errors.append(error)

    print(f"\nDevices: {num_devices}")
    print("True Global Mean:", true_global)
    print("DP Global Mean  :", dp_global)
    print("Error (L2 norm) :", error)

plt.figure(figsize=(8, 5))
plt.plot(device_counts, errors, marker='o')
plt.title("DP Error vs Number of Devices")
plt.xlabel("Number of Devices")
plt.ylabel("Error (L2 norm)")
plt.grid(True)
plt.tight_layout()
plt.show()
