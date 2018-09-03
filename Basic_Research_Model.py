import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
np.random.seed(0)

class Basic_Research_Model(object):
    def __init__(self, input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function):

        self.layer_0_dim                  = input_dim
        self.layer_1_dim                  = input_dim * 100
        self.layer_2_dim                  = input_dim * 100
        self.layer_3_dim                  = output_dim

        self.alpha                        = alpha
        if (pruning != "True") & (stabilizing!= "True"):
            self.epochs                       = epochs * 3
        elif (pruning == "True") & (stabilizing == "True"):
            self.epochs                       = epochs
        else:
            self.epochs                       = epochs * 2
        self.beta                         = beta
        self.rounds                       = rounds
        self.pruning                      = pruning
        self.pruning_criterion            = pruning_criterion
        self.stabilizing                  = stabilizing
        self.stabilizing_criterion        = stabilizing_criterion
        self.randomness                   = randomness
        self.function                     = function

        self.synapse_layer_0_to_layer_1   ,\
        self.synapse_layer_1_to_layer_2   ,\
        self.synapse_layer_2_to_layer_3   = self.initialize_weights()

        self.synapse_layer_0_to_layer_1_update          = np.zeros_like( self.synapse_layer_0_to_layer_1 )
        self.synapse_layer_1_to_layer_2_update          = np.zeros_like( self.synapse_layer_1_to_layer_2 )
        self.synapse_layer_2_to_layer_3_update          = np.zeros_like( self.synapse_layer_2_to_layer_3 )


    def initialize_weights(self):

        synapse_layer_0_to_layer_1                                              = (np.random.random((self.layer_0_dim                                 , self.layer_1_dim                           )) -0.5 ) * 0.5
        synapse_layer_1_to_layer_2                                              = (np.random.random((self.layer_1_dim                                 , self.layer_2_dim                           )) -0.5 ) * 0.5
        synapse_layer_2_to_layer_3                                              = (np.random.random((self.layer_2_dim                                 , self.layer_3_dim                           )) -0.5 ) * 0.5

        return  synapse_layer_0_to_layer_1,\
                synapse_layer_1_to_layer_2,\
                synapse_layer_2_to_layer_3

    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def processor(self, x):
        if self.function == "sigmoid":
            output = 1 / (1 + np.exp(-1 * x))
        if self.function == "ReLu":
            output = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                if (x[i] >= 0) == True & (x[i] <= 1) == True:
                    output[i] = x[i]
                if (x[i]) > 1 == True:
                    output[i] = 1
                if (x[i]) < 0 == True:
                    output[i] = 0
        return output

    def processor_output_to_derivative(self, output):
        if self.function == "sigmoid":
            output = output * (1 - output) * 1
        if self.function == "ReLu":
            dummy = np.zeros(output.shape[0])
            for i in range(output.shape[0]):
                if (output[i] > 0) == True & (output[i] < 1) == True:
                    dummy[i] = 1
                if (output[i] >= 1) == True:
                    dummy[i] = 0
                if (output[i] <= 0) == True:
                    dummy[i] = 0
            output = dummy
        return output

    def generate_values_for_each_layer(self, input):

        layer_0                   = copy.deepcopy(np.array(input))

        layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) )

        layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )

        layer_3                   = self.sigmoid(np.dot(layer_2                          , self.synapse_layer_2_to_layer_3                                                          ) )

        return   layer_0             ,\
                    layer_1             ,\
                    layer_2             ,\
                    layer_3

    def train_for_each(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       output):

        layer_3_error                           = np.array([output - layer_3])

        layer_3_delta                           = (layer_3_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_3)])

        layer_2_delta                           = (layer_3_delta.dot(self.synapse_layer_2_to_layer_3.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        self.synapse_layer_0_to_layer_1_update                            += np.atleast_2d(layer_0                                                          ).T.dot(layer_1_delta                              )
        self.synapse_layer_1_to_layer_2_update                            += np.atleast_2d(layer_1                                                          ).T.dot(layer_2_delta                              )
        self.synapse_layer_2_to_layer_3_update                            += np.atleast_2d(layer_2                                                          ).T.dot(layer_3_delta                              )

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha
        self.synapse_layer_2_to_layer_3                                   += self.synapse_layer_2_to_layer_3_update                   * self.alpha

        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0
        self.synapse_layer_2_to_layer_3_update                            *= 0

    def train_for_each_after_pruning(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       output):

        layer_3_error                           = np.array([output - layer_3])

        layer_3_delta                           = (layer_3_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_3)])

        layer_2_delta                           = (layer_3_delta.dot(self.synapse_layer_2_to_layer_3.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        self.synapse_layer_0_to_layer_1_update                            += np.atleast_2d(layer_0                                                          ).T.dot(layer_1_delta                              ) * self.synapse_layer_0_to_layer_1_diagram
        self.synapse_layer_1_to_layer_2_update                            += np.atleast_2d(layer_1                                                          ).T.dot(layer_2_delta                              ) * self.synapse_layer_1_to_layer_2_diagram
        self.synapse_layer_2_to_layer_3_update                            += np.atleast_2d(layer_2                                                          ).T.dot(layer_3_delta                              ) * self.synapse_layer_2_to_layer_3_diagram

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha
        self.synapse_layer_2_to_layer_3                                   += self.synapse_layer_2_to_layer_3_update                   * self.alpha

        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0
        self.synapse_layer_2_to_layer_3_update                            *= 0

    def synapse_pruning_diagram(self, matrix, criterion):
        diagram = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i][j]) >= criterion:
                    diagram[i][j] = 1
        return diagram

    def synapse_stabilizing(self, matrix, criterion):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.abs(matrix[i][j]) >= criterion:
                    matrix[i][j] = criterion * ( matrix[i][j] / np.abs(matrix[i][j]))
        return matrix

    def fit(self, input_list, output_list):

        for i in range(self.epochs):

            random_index = np.random.randint(input_list.shape[0])
            input        = input_list[random_index]
            output       = output_list[random_index]

            layer_0,\
            layer_1, \
            layer_2, \
            layer_3  = self.generate_values_for_each_layer(input)

            self.train_for_each(
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       output)

        if self.pruning == "True":

            self.synapse_layer_0_to_layer_1_diagram                             = self.synapse_pruning_diagram(self.synapse_layer_0_to_layer_1                          , self.pruning_criterion)
            self.synapse_layer_1_to_layer_2_diagram                             = self.synapse_pruning_diagram(self.synapse_layer_1_to_layer_2                          , self.pruning_criterion)
            self.synapse_layer_2_to_layer_3_diagram                             = self.synapse_pruning_diagram(self.synapse_layer_2_to_layer_3                          , self.pruning_criterion)

            self.synapse_layer_0_to_layer_1                                    *= self.synapse_layer_0_to_layer_1_diagram
            self.synapse_layer_1_to_layer_2                                    *= self.synapse_layer_1_to_layer_2_diagram
            self.synapse_layer_2_to_layer_3                                    *= self.synapse_layer_2_to_layer_3_diagram

            for i in range(self.epochs):

                random_index = np.random.randint(input_list.shape[0])
                input        = input_list[random_index]
                output       = output_list[random_index]

                layer_0,\
                layer_1, \
                layer_2, \
                layer_3  = self.generate_values_for_each_layer(input)

                self.train_for_each_after_pruning(
                           layer_0,
                           layer_1,
                           layer_2,
                           layer_3,
                           output)

        if self.stabilizing == "True":

            for i in range(self.epochs):

                self.synapse_layer_0_to_layer_1                             = self.synapse_stabilizing(self.synapse_layer_0_to_layer_1                          , self.stabilizing_criterion)
                self.synapse_layer_1_to_layer_2                             = self.synapse_stabilizing(self.synapse_layer_1_to_layer_2                          , self.stabilizing_criterion)
                self.synapse_layer_2_to_layer_3                             = self.synapse_stabilizing(self.synapse_layer_2_to_layer_3                          , self.stabilizing_criterion)

                random_index = np.random.randint(input_list.shape[0])
                input        = input_list[random_index]
                output       = output_list[random_index]

                layer_0,\
                layer_1, \
                layer_2, \
                layer_3  = self.generate_values_for_each_layer(input)

                self.train_for_each_after_pruning(
                           layer_0,
                           layer_1,
                           layer_2,
                           layer_3,
                           output)

            self.synapse_layer_0_to_layer_1                             = self.synapse_stabilizing(self.synapse_layer_0_to_layer_1                          , self.stabilizing_criterion)
            self.synapse_layer_1_to_layer_2                             = self.synapse_stabilizing(self.synapse_layer_1_to_layer_2                          , self.stabilizing_criterion)
            self.synapse_layer_2_to_layer_3                             = self.synapse_stabilizing(self.synapse_layer_2_to_layer_3                          , self.stabilizing_criterion)

        return self



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



    def train_for_random_input_inner_player_A(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       output):

        layer_3_error                           = np.array([output - layer_3])

        layer_3_delta                           = (layer_3_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_3)])

        layer_2_delta                           = (layer_3_delta.dot(self.synapse_layer_2_to_layer_3.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * np.array([self.processor_output_to_derivative(layer_0)])

        if self.randomness == "True":
            self.random_input_inner_for_player_A[self.random_target] += layer_0_delta[0][0:self.random_input_inner_for_player_A.shape[0]][self.random_target] * self.beta
        else:
            self.random_input_inner_for_player_A += layer_0_delta[0][0:self.random_input_inner_for_player_A.shape[0]] * self.beta




    def train_for_random_input_inner_player_B(self,
                       layer_0,
                       layer_1,
                       layer_2,
                       layer_3,
                       output):

        layer_3_error                           = np.array([output - layer_3])

        layer_3_delta                           = (layer_3_error                                                                                            ) * np.array([self.sigmoid_output_to_derivative(layer_3)])

        layer_2_delta                           = (layer_3_delta.dot(self.synapse_layer_2_to_layer_3.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_2)])

        layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) ) * np.array([self.sigmoid_output_to_derivative(layer_1)])

        layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * np.array([self.processor_output_to_derivative(layer_0)])

        if self.randomness == "True":
            self.random_input_inner_for_player_B[self.random_target] += layer_0_delta[0][-self.random_input_inner_for_player_A.shape[0]: ][self.random_target] * self.beta
        else:
            self.random_input_inner_for_player_B += layer_0_delta[0][-self.random_input_inner_for_player_B.shape[0]: ] * self.beta




    def deduct_from(self, random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B):

        self.random_input_inner_for_player_A = random_input_inner_for_player_A
        self.random_input_inner_for_player_B = random_input_inner_for_player_B

        for i in range(self.rounds):

            #---------------------A-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)

            self.random_target      = np.random.randint(self.random_input_inner_for_player_A.shape[0])

            layer_0,\
            layer_1, \
            layer_2, \
            layer_3  = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_A    = copy.deepcopy(layer_3)
            desired_output_for_player_A[0] = desired_payoff_for_player_A
            #desired_output_for_player_A[1] = 0

            self.train_for_random_input_inner_player_A(
                                 layer_0,
                                 layer_1,
                                 layer_2,
                                 layer_3,
                                 desired_output_for_player_A)

            #---------------------B-----------------------

            self.random_input_for_player_A       = self.processor(self.random_input_inner_for_player_A)
            self.random_input_for_player_B       = self.processor(self.random_input_inner_for_player_B)

            self.random_target      = np.random.randint(self.random_input_inner_for_player_B.shape[0])

            layer_0,\
            layer_1, \
            layer_2, \
            layer_3  = self.generate_values_for_each_layer(np.concatenate(( self.random_input_for_player_A,  self.random_input_for_player_B)))

            desired_output_for_player_B    = copy.deepcopy(layer_3)
            desired_output_for_player_B[1] = desired_payoff_for_player_B
            #desired_output_for_player_B[0] = 0

            self.train_for_random_input_inner_player_B(
                                 layer_0,
                                 layer_1,
                                 layer_2,
                                 layer_3,
                                 desired_output_for_player_B)



            if i % 100 == 0:
                print(np.round(self.processor(self.random_input_inner_for_player_A), 2))
                print(np.round(self.processor(self.random_input_inner_for_player_B), 2))
                print(np.round(layer_3, 2))

        return self.random_input_inner_for_player_A, self.random_input_inner_for_player_B

