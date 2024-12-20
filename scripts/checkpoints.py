import os

# Get the list of all files and directories
path = ".//models/Ensertainty_ensemble_MNIST_3class/"
dir_list = os.listdir(path)

# find all the files in the directory that match MNIST_EncoderCoupleRQS_2dim in prefix and 2classes in suffix

# print the list
models = []
models_eqx = []
states = []
for i in dir_list:
    if i.startswith("mnist") and i.endswith("_2dim"):
        models.append(f"{path}{i}/")
        models_eqx.append("model.eqx")
        states.append("state.pickle")

print("\n")
print(models)
print("\n")
print(models_eqx)
print("\n")
print(states)

print(len(models), len(models_eqx), len(states))