import numpy as np
import os
import matplotlib.pyplot as plt
from threading import Thread
from json import JSONEncoder
import json


"""
nn_architecture = [
    {"input_dim": 2, "output_dim": 4, "activation": "self.relu"},
    {"input_dim": 4, "output_dim": 6, "activation": "self.relu"},
    {"input_dim": 6, "output_dim": 6, "activation": "self.relu"},
    {"input_dim": 6, "output_dim": 4, "activation": "self.relu"},
    {"input_dim": 4, "output_dim": 1, "activation": "self.sigmoid"},
]

example:

    nn_architecture = [
        {"input_dim": 30, "output_dim": 40, "activation": "sigmoid"},
        {"input_dim": 40, "output_dim": 10, "activation": "sigmoid"},
        {"input_dim": 10, "output_dim": 1, "activation": "sigmoid"},
    ]

input_x and output_y is in array format    
 input_x [[sample 0 input 0,sample 1 input 0,sample 2 input 0,...],
    [[sample 0 input 1,sample 1 input 1,sample 2 input 1,...],
    [[sample 0 input 2,sample 1 input 2,sample 2 input 2,...],...]

 output_y [[sample 0 output 0,sample 1 output 0,sample 2 output 0,...],
    [[sample 0 output 1,sample 1 output 1,sample 2 output 1,...],
    [[sample 0 output 2,sample 1 output 2,sample 2 output 2,...],...]

example:

    input_x = np.array([[100,99,98,1,0],[97,98,96,0,1]])
    output_y = np.array([0.1,0.1,0.1, 0.9,0.9])

usage:
    bp_ann = BpAnn(nn_architecture=nn_architecture, seed=99)
    bp_ann.train(X=input_x,Y=output_y,epochs=50000,learning_rate=0.3, title='Something')
    bp_ann.test(X=input_x, Y=output_y)
    
"""

keep_going = True

def key_capture_thread():
    global keep_going
    input()
    keep_going = False

class BpAnn:

    def __init__(self, model_name, nn_architecture, seed=99):

        np.random.seed(seed)
        number_of_layers = len(nn_architecture)
        self.params_values = {}

        self.params_values_show = np.array([], dtype=np.float32)
        self.param_file_name = model_name + '_parameters.txt'
        self.model_name = model_name

        param_saved = self.read_nn_parameters()
        print("Neural network structure {}\n".format(nn_architecture))

        if param_saved:
            print("Found saved parameter. Loading ...")
            if param_saved['nn_architecture'] != nn_architecture:
                print("Saved parameter Neural network structure {}\n".format(param_saved['nn_architecture']))
                raise ValueError("Expected Neural network structure is different from saved parameter, exited NN\n")
            else:
                self.current_epoch = int(param_saved['current_epoch'])
                print("Current epoch {}\n".format(self.current_epoch))
                for key in param_saved['nn_parameters']:
                    self.params_values[key] = np.array(param_saved['nn_parameters'][key], dtype=np.float32)
                print("Parameters loaded")
        else:
            self.current_epoch = -1
            print("Not found saved parameter. Creating new parameters ...")

        for idx, layer in enumerate(nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]

            if not param_saved:
                self.params_values['W' + str(layer_idx)] = np.random.randn(
                    layer_output_size, layer_input_size).astype(np.float32) * 0.1
                self.params_values['b' + str(layer_idx)] = np.random.randn(
                    layer_output_size, 1).astype(np.float32) * 0.1

            self.params_values_show = np.concatenate((self.params_values_show,self.params_values['W' + str(layer_idx)].flatten()))
            self.params_values_show = np.concatenate((self.params_values_show,self.params_values['b' + str(layer_idx)].flatten()))

        self.nn_architecture = nn_architecture
        self.accuracy = None


    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0,Z)

    def sigmoid_backward(self, dA, Z):
        sig = self.sigmoid(Z)
        return dA * sig * (1 - sig)

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True, dtype=np.float32)
        dZ[Z <= 0] = 0
        return dZ

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="sigmoid"):

        Z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation == "relu":
            activation_func = self.relu
        elif activation == "sigmoid":
            activation_func = self.sigmoid
        else:
            raise Exception('Non-supported activation function')

        return activation_func(Z_curr), Z_curr

#input_X is the input set as an array
    def full_forward_propagation(self, input_X):

        memory = {}
        A_curr = input_X

        for idx, layer in enumerate(self.nn_architecture):

            layer_idx = idx + 1
            A_prev = A_curr

            activ_function_curr = layer["activation"]
            W_curr = self.params_values["W" + str(layer_idx)]
            b_curr = self.params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        return A_curr, memory

    def get_cost_value(self,Y_hat, Y):
        m = Y_hat.shape[1]
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
        return np.squeeze(cost)

    def convert_prob_into_class(self, AL):
        pred = np.copy(AL)
        pred[AL > 0.5] = 1
        pred[AL <= 0.5] = 0
        return pred

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = self.convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def single_layer_backward_propagation(self, dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="sigmoid"):
        m = A_prev.shape[1]

        if activation == "relu":
            backward_activation_func = self.relu_backward
        elif activation == "sigmoid":
            backward_activation_func = self.sigmoid_backward
        else:
            raise Exception('Non-supported activation function')

        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr

    def full_backward_propagation(self, Y_hat, Y, memory):
        grads_values = {}
        # m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)

        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]

            dA_curr = dA_prev

            A_prev = memory["A" + str(layer_idx_prev)]
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = self.params_values["W" + str(layer_idx_curr)]
            b_curr = self.params_values["b" + str(layer_idx_curr)]

            dA_prev, dW_curr, db_curr = self.single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr

        return grads_values

    def update(self, grads_values, learning_rate):
        self.params_values_show = np.array([], dtype=np.float32)
        for layer_idx, layer in enumerate(self.nn_architecture):
            self.params_values["W" + str(layer_idx+1)] -= learning_rate * grads_values["dW" + str(layer_idx+1)]
            self.params_values["b" + str(layer_idx+1)] -= learning_rate * grads_values["db" + str(layer_idx+1)]
            self.params_values_show = np.concatenate((self.params_values_show,self.params_values['W' + str(layer_idx+1)].flatten()))
            self.params_values_show = np.concatenate((self.params_values_show,self.params_values['b' + str(layer_idx+1)].flatten()))

    def train(self, X, Y, epochs, learning_rate, title):

        #show_plot = ShowPlot(title='Train result', xlabel='Samples', ylabel='Accuracy of output', shape=self.params_values_show.shape[0])
        show_plot = ShowPlot(title=title, xlabel='Samples', ylabel='Accuracy of output', shape=Y.shape[0])

        np.set_printoptions(precision=3, suppress=True)

        Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()

        save_param_interval = 10

        for i in range(self.current_epoch+1, epochs):
            Y_hat, cashe = self.full_forward_propagation(X)
#            cost = self.get_cost_value(Y_hat, Y)
#            cost_history.append(cost)
            t_accuracy = self.get_accuracy_value(Y_hat, Y)

            self.accuracy = Y-Y_hat
            #print(self.accuracy)
            #input("\nPress any key to continue ...")
            print("Epoch : " + str(i))
            print("t_accuracy: {}".format(t_accuracy))

            #if i % 100 == 0:
            show_plot.draw_train_result(self.accuracy[0])
#                show_plot.draw_train_result(self.params_values_show)

            grads_values = self.full_backward_propagation(Y_hat, Y, cashe)
            self.update(grads_values, learning_rate)

            if i % save_param_interval == 0:
                self.save_nn_parameters(current_epoch=str(i))
                print("Parameter saved")

            if not keep_going:
                break

        print("\nTraining finished. Press a key\n")
        self.save_nn_parameters(current_epoch=str(i))
        input("\nParameter saved. Press any key to exit ...")

    #        return cost_history, accuracy_history

    def test(self, X, Y):

#        show_plot = ShowPlot(title='Test result', shape=Y.shape[0])
        np.set_printoptions(precision=3, suppress=True)

        Y_hat, cashe = self.full_forward_propagation(X)


        filter_arr = []

# if the element is higher than 0.25 and less than 0.75, set the value to False, otherwise True:
        refuse_count = 0
        correct_one = 0
        wrong_one = 0
        correct_zero = 0
        wrong_zero = 0

        for idy, y_h in np.ndenumerate(Y_hat[0]):
            if y_h <= 0.2:
                filter_arr.append(True)
                Y_hat[0][idy] = 0
                if Y[idy] == 0:
                    correct_zero += 1
                else:
                    wrong_zero += 1

            elif y_h >= 0.8:
                filter_arr.append(True)
                Y_hat[0][idy] = 1
                if Y[idy] == 1:
                    correct_one += 1
                else:
                    wrong_one += 1
            else:
                filter_arr.append(False)
                refuse_count += 1

        new_Y_hat = Y_hat[0][filter_arr]
        new_Y = Y[filter_arr]

        refuse_rate = refuse_count / Y.shape[0]
#        test_result = Y_hat[0]//0.5
#        show_plot.draw_test_result(Y,test_result)

#        plt.show()

#        correct_rate = 1 - np.sum(np.abs(Y - test_result))/Y.shape[0]
        correct_rate = 1 - np.sum(np.abs(new_Y - new_Y_hat))/new_Y.shape[0]

        print("\n\nRefuse rate:  " + str(refuse_rate))
        print("\n\nCorrect rate:  " + str(correct_rate))

        print("\n\nCorrect one  " + str (correct_one))
        print("\n\nWrong one  " + str(wrong_one))
        print("\n\nCorrect zero  " + str(correct_zero))
        print("\n\nWrong zero  " + str(wrong_zero))

        print("\n\nExpected result after removing refused\n")
        print(new_Y)
        print("\n\nActual result after removing refused\n")
        print(new_Y_hat)

    def get_weights(self):
        return self.params_values

    def save_nn_parameters(self, current_epoch):
        repo_data = {}
        repo_data['model_name'] = self.model_name
        repo_data['current_epoch'] = current_epoch
        repo_data['nn_architecture'] = self.nn_architecture

        params_values = {}

        for key in self.params_values.keys():
            params_values[key] = self.params_values[key].tolist()

        repo_data['nn_parameters'] = params_values

        file_path = os.path.join('.\\nn_model_repo', self.param_file_name)

        with open(file_path, 'w') as fp:
            json.dump(repo_data, fp)

        fp.close()

    # this is called by init automatically to initialize parameter
    def read_nn_parameters(self):
        file_path = os.path.join('.\\nn_model_repo', self.param_file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as fp:
                repo_data = json.load(fp)
            fp.close()
            return repo_data
        else:
            return None

class ShowPlot:

    def __init__(self, title,xlabel, ylabel, shape):
        self.ax = plt.axes()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        plt.title(title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.ylim(-1, 1)
        self.line_zero = np.zeros(shape)
        self.plot_x = np.arange(1,shape+1)

    def draw_train_result(self, plot_y):
        self.ax.clear()
        plt.ylim(-1, 1)
        self.ax.plot(self.plot_x, self.line_zero)
        self.ax.plot(self.plot_x, plot_y)
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.draw()
        plt.pause(0.01)

    def draw_test_result(self, ploy_y, plot_y_hat):
        plt.ylim(0, 1)
        self.ax.plot(self.plot_x, ploy_y)
        self.ax.plot(self.plot_x, plot_y_hat)
        plt.draw()
        plt.pause(0.01)






