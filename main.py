import random
import numpy as np
import numdifftools as nd
from datetime import datetime

class Network:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights, self.biases = [], []
        r_normal = np.random.randn

        B = layer_sizes[1:]
        biases = [r_normal(b, 1) for b in B]
        W = zip(layer_sizes[:-1], layer_sizes[1:])
        weights = [r_normal(b, a) for a, b in W]

        for np_w_mat in weights:
            w_mat = []
            for i in range(len(np_w_mat)):
                w_mat.append([])
                for j in range(len(np_w_mat[i])):
                    w_mat[i].append(np_w_mat[i][j])
            self.weights.append(w_mat)

        for np_b_vec in biases:
            b_vec = []
            for k in range(len(np_b_vec)):
                b_vec.append(np_b_vec[k][0])
            self.biases.append(b_vec)


    def apply_activation_function(layer_num, z_vec):
        def activation_function(z):
            activation = 1 / (1 + np.exp(-z))
            return activation
        activated_vec = z_vec
        for n in range(len(activated_vec)):
            activate = activation_function
            activated_vec[n] = activate(z_vec[n])
        return activated_vec


    def vector_scaling(vector, factor):
        for n in range(len(vector)):
            vector[n] *= factor
        return vector


    def vector_addition(vec1, vec2):
        for n in range(len(vec1)):
            vec1[n] += vec2[n]
        return vec1


    def matrix_on_vector(matrix, vector):
        transformed = []
        for n in range(len(matrix)):
            sum = 0
            for m in range(len(matrix[n])):
                sum += matrix[n][m] * vector[m]
            transformed.append(sum)
        return transformed


    def unpack_matrix(matrix):
        unpacked = []
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                unpacked.append(matrix[i][j])
        return unpacked


    def pack_weight_bias_vector(self):
        weights, biases = self.weights, self.biases
        unpack_matrix = Network.unpack_matrix
        weight_bias_vector = []
        for layer_num in range(len(self.layer_sizes) - 1):
            hash_w_b = {0: weights[layer_num], 1: biases[layer_num]}
            hash_unpack = {0: unpack_matrix, 1: lambda x: x}
            for alternate in range(2):
                data_obj = hash_w_b[alternate]
                unpack = hash_unpack[alternate]
                sub_vector = unpack(data_obj)
                weight_bias_vector += sub_vector
        return weight_bias_vector


    def unpack_weight_bias_vector(self, weight_bias_vector):
        weights, biases, unpack_idx = [], [], 0
        for layer_num in range(len(self.layer_sizes) - 1):
            w_mat_num_rows = len(self.weights[layer_num])
            w_mat_num_cols = len(self.weights[layer_num][0])
            weight_matrix, bias_vector = [], []
            for i in range(w_mat_num_rows):
                weight_matrix.append([])
                for j in range(w_mat_num_cols):
                    w = weight_bias_vector[unpack_idx]
                    weight_matrix[i].append(w)
                    unpack_idx += 1
            weights.append(weight_matrix)
            len_b_vec = len(self.biases[layer_num])
            for k in range(len_b_vec):
                b = weight_bias_vector[unpack_idx]
                bias_vector.append(b)
                unpack_idx += 1
            biases.append(bias_vector)
        return weights, biases


    def make_cost_function(self, data_point, correct_output):
        def cost_function(w_b_parameters_vec):
            unpack_w_b = Network.unpack_weight_bias_vector
            weights, biases = unpack_w_b(self, w_b_parameters_vec)
            activate = Network.apply_activation_function
            apply_mat = Network.matrix_on_vector
            add_vec = Network.vector_addition
            layer_activations = data_point
            number_layers = len(self.layer_sizes)
            for layer_index in range(number_layers - 1):
                weight_mat = weights[layer_index]
                bias_vec = biases[layer_index]
                weigthed_activations = apply_mat(weight_mat, layer_activations)
                z_vec = add_vec(weigthed_activations, bias_vec)
                layer_activations = activate(layer_index, z_vec)
            actual_output = layer_activations
            sum_squared_errors = 0
            for k in range(len(actual_output)):
                error = actual_output[k] - correct_output[k]
                sum_squared_errors += error**2
            cost_for_data_point = sum_squared_errors
            return cost_for_data_point
        return cost_function


    def update_weights_biases(self, new_w_b_vector):
        unpack_w_b = Network.unpack_weight_bias_vector
        new_weights, new_biases = unpack_w_b(self, new_w_b_vector)
        for index_W in range(len(self.weights)):
            self.weights[index_W] = new_weights[index_W]
        for index_B in range(len(self.biases)):
            self.biases[index_B] = new_biases[index_B]


    def gradient_descent(initial_w_b_vec, cost_func):
        cost_gradient = nd.Gradient(cost_func)
        add_vec = Network.vector_addition
        scale = Network.vector_scaling
        GD_w_b_vec = initial_w_b_vec
        learning_rate = 1
        for _ in range(9999):
            step = scale(cost_gradient(GD_w_b_vec), -1)
            sized_step = scale(step, learning_rate)
            GD_w_b_vec = add_vec(GD_w_b_vec, sized_step)
        after_GD_w_b_vec = GD_w_b_vec
        return after_GD_w_b_vec


    def train(self, batch_size, training_data_matrix, correct_output_matrix):
        GD = Network.gradient_descent
        add_vec, scale = Network.vector_addition, Network.vector_scaling
        for batch_num in range(int(len(training_data_matrix)/batch_size)):
            a, b = batch_num * batch_size, (batch_num + 1) * batch_size
            num_w_b_parameters = len(self.pack_weight_bias_vector())
            sum_w_b_deltas_vec = [0 for _ in range(num_w_b_parameters)]
            for index_in_batch in range(a, b):
                data_point = training_data_matrix[index_in_batch]
                correct_output = correct_output_matrix[index_in_batch]
                cost_at_dp = self.make_cost_function(data_point, correct_output)
                initial_w_b_vec = self.pack_weight_bias_vector()
                after_GD_w_b_vec = GD(initial_w_b_vec, cost_at_dp)
                neg_initial_w_b_vec = scale(initial_w_b_vec, -1)
                delta_w_b_vec = add_vec(after_GD_w_b_vec, neg_initial_w_b_vec)
                sum_w_b_deltas_vec = add_vec(sum_w_b_deltas_vec, delta_w_b_vec)
            batch_delta_w_b_vec = scale(sum_w_b_deltas_vec, 1 / batch_size)
            current_w_b_vec = self.pack_weight_bias_vector()
            updated_w_b_vec = add_vec(current_w_b_vec, batch_delta_w_b_vec)
            self.update_weights_biases(updated_w_b_vec)


    def transition_function(self, from_layer, layer_activations):
        activate = Network.apply_activation_function
        add_vec = Network.vector_addition
        apply_mat = Network.matrix_on_vector
        weights = self.weights[from_layer]
        biases = self.biases[from_layer]
        weigthed_activations = apply_mat(weights, layer_activations)
        z = add_vec(weigthed_activations, biases)
        next_layer_activations = activate(from_layer, z)
        return next_layer_activations

    #follows form Y = F(X)
    def evaluate(self, X):
        layer_activations = X
        fn = self.transition_function
        for l in range(len(self.layer_sizes) - 1):
            layer_activations = fn(l, layer_activations)
        Y = layer_activations
        return Y