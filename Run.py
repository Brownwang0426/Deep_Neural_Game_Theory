import numpy as np
import scipy as sp
import random
import copy
import matplotlib.pyplot as plt
np.random.seed(0)



input_list = []

input_0 = np.array([1, 0, 0, 1, 0, 0])
input_list.append(input_0)
input_0 = np.array([0, 1, 0, 1, 0, 0])
input_list.append(input_0)
input_0 = np.array([0, 0, 1, 1, 0, 0])
input_list.append(input_0)
input_0 = np.array([1, 0, 0, 0, 1, 0])
input_list.append(input_0)
input_0 = np.array([0, 1, 0, 0, 1, 0])
input_list.append(input_0)
input_0 = np.array([0, 0, 1, 0, 1, 0])
input_list.append(input_0)
input_0 = np.array([1, 0, 0, 0, 0, 1])
input_list.append(input_0)
input_0 = np.array([0, 1, 0, 0, 0, 1])
input_list.append(input_0)
input_0 = np.array([0, 0, 1, 0, 0, 1])
input_list.append(input_0)



output_list = []

output_0 = np.array([0.7, 0.3])
output_list.append(output_0)
output_0 = np.array([0.3, 0.3])
output_list.append(output_0)
output_0 = np.array([0.3, 0.3])
output_list.append(output_0)
output_0 = np.array([0.6, 0.7])
output_list.append(output_0)
output_0 = np.array([0.5, 0.5])
output_list.append(output_0)
output_0 = np.array([0.2, 0.7])
output_list.append(output_0)
output_0 = np.array([0.8, 0.0])
output_list.append(output_0)
output_0 = np.array([0.7, 0.3])
output_list.append(output_0)
output_0 = np.array([0.3, 0.9])
output_list.append(output_0)



input_list  = np.array(input_list)
output_list = np.array(output_list)

#------------------------------------------IMPORTING MODEL------------------------------------------------------

from Basic_Research_Model import *

input_dim             = input_list.shape[1]
output_dim            = output_list.shape[1]

alpha                 = 0.1
epochs                = 1000                                                                                       #---------------------------

beta                  = 0.1
rounds                = 1000                                                                                      #---------------------------

pruning               = "not True"                                                                                      #---------------------------
pruning_criterion     = 0.225                                                                                           #---------------------------
stabilizing           = "not True"                                                                                         #---------------------------
stabilizing_criterion = 1                                                                                           #---------------------------

randomness            = "not True"                                                                                      #---------------------------

function              = "sigmoid"

Machine               = Basic_Research_Model(input_dim, output_dim, alpha, epochs, beta, rounds, pruning, pruning_criterion, stabilizing, stabilizing_criterion, randomness, function)






Machine.fit(input_list, output_list)




#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


strategy_neurons_size             = 3
random_input_inner_for_player_A   = (np.random.random(strategy_neurons_size) - 0.5 ) * 0.1 - 0
desired_payoff_for_player_A       = 1
random_input_inner_for_player_B   = (np.random.random(strategy_neurons_size) - 0.5 ) * 0.1 - 0
desired_payoff_for_player_B       = 1


print(np.round(Machine.processor(random_input_inner_for_player_A), 2))
print(np.round(Machine.processor(random_input_inner_for_player_B), 2))

random_input_inner_for_player_A, random_input_inner_for_player_B = Machine.deduct_from(random_input_inner_for_player_A, random_input_inner_for_player_B, desired_payoff_for_player_A, desired_payoff_for_player_B)

print(np.round(Machine.processor(random_input_inner_for_player_A), 2))
print(np.round(Machine.processor(random_input_inner_for_player_B), 2))
