import numpy as np
from bp_ann import BpAnn

# simple XOR test case
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
]

# input data
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
], dtype=np.float32)

# expected output
Y = np.array([0, 1, 1, 0], dtype=np.float32)

# build and train model
model = BpAnn(model_name="xor_demo", nn_architecture=nn_architecture, seed=42)
model.train(X=X, Y=Y, epochs=3000, learning_rate=0.1, title="XOR training")
model.test(X=X, Y=Y)
