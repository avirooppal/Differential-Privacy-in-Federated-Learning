import numpy as np

device1_data = np.random.rand(100,2)
device2_data = np.random.rand(100,2)
print(device1_data)
print(device2_data)


def local_train(data):
    return np.mean(data,axis=0)


def add_dp_noise(update,epsilon=1.0,sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0,scale,size=update.shape)
    print("update + noise",update + noise)
    return update + noise 


update1 = local_train(device1_data)
update2 = local_train(device2_data)
print("Update1",update1)
print("Update2",update2)

dp_update1 = add_dp_noise(update1)
dp_update2 = add_dp_noise(update2)
print("dp update1",dp_update1)
print("dp update2",dp_update2)
 

global_model = (dp_update1 + dp_update2) / 2

print("Global model:", global_model)
