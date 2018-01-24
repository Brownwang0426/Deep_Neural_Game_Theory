import copy
import numpy as np

np.random.seed(0)

class Game_Net(object):
    def __init__(self, middle_dim, alpha = 0.01, beta = 0.01, epochs = 1000):

        self.layer_memory_start_1_dim     = middle_dim + 20
        self.layer_memory_end_1_dim       = middle_dim + 20
        self.layer_0_dim                  = middle_dim
        self.layer_1_dim                  = middle_dim + 20
        self.layer_2_dim                  = 2                  # <------------------
        self.layer_write_1_dim            = middle_dim + 20
        self.layer_read_1_dim             = middle_dim + 20

        self.alpha                        = alpha
        self.beta                         = beta
        self.epochs                       = epochs

        self.synapse_layer_0_to_layer_1                                         ,\
        self.synapse_layer_1_to_layer_2                                         ,\
        self.synapse_layer_1_to_layer_write_1                                   ,\
        self.synapse_layer_write_1_to_layer_memory_end_1                        ,\
        self.synapse_layer_memory_start_1_to_layer_memory_end_1                 ,\
        self.synapse_layer_memory_end_1_to_layer_memory_start_1                 ,\
        self.synapse_layer_1_to_future_layer_1                                  ,\
        self.synapse_layer_1_to_future_layer_read_1                             ,\
        self.layer_write_1_size_controller                                      ,\
        self.layer_read_1_enhancer                                              ,\
        self.layer_writer_1_enhancer                                            ,\
        self.tanh_enhancer                                                      = self.initialize_weights()

        self.synapse_layer_0_to_layer_1_update                                  = np.zeros_like(self.synapse_layer_0_to_layer_1                          )
        self.synapse_layer_1_to_layer_2_update                                  = np.zeros_like(self.synapse_layer_1_to_layer_2                          )
        self.synapse_layer_1_to_layer_write_1_update                            = np.zeros_like(self.synapse_layer_1_to_layer_write_1                    )
        self.synapse_layer_write_1_to_layer_memory_end_1_update                 = np.zeros_like(self.synapse_layer_write_1_to_layer_memory_end_1         )
        self.synapse_layer_memory_start_1_to_layer_memory_end_1_update          = np.zeros_like(self.synapse_layer_memory_start_1_to_layer_memory_end_1  )
        self.synapse_layer_memory_end_1_to_layer_memory_start_1_update          = np.zeros_like(self.synapse_layer_memory_end_1_to_layer_memory_start_1  )
        self.synapse_layer_1_to_future_layer_read_1_update                      = np.zeros_like(self.synapse_layer_1_to_future_layer_read_1              )
        self.synapse_layer_1_to_future_layer_1_update                           = np.zeros_like(self.synapse_layer_1_to_future_layer_1                   )
        self.layer_write_1_size_controller_update                               = np.zeros_like(self.layer_write_1_size_controller                       )
        self.layer_read_1_enhancer_update                                       = np.zeros_like(self.layer_read_1_enhancer                               )
        self.layer_writer_1_enhancer_update                                     = np.zeros_like(self.layer_writer_1_enhancer                             )
        self.tanh_enhancer_update                                               = np.zeros_like(self.tanh_enhancer                                       )

        self.A_1                                = 0
        self.B_2                                = 0
        self.A_3                                = 0
        self.B_4                                = 0
        self.A_5                                = 0
        self.B_6                                = 0
        self.A_7                                = 0
        self.B_8                                = 0
        self.A_9                                = 0

        self.A_1_Outcome                        = 0
        self.B_2_Outcome                        = 0
        self.A_3_Outcome                        = 0
        self.B_4_Outcome                        = 0
        self.A_5_Outcome                        = 0
        self.B_6_Outcome                        = 0
        self.A_7_Outcome                        = 0
        self.B_8_Outcome                        = 0
        self.A_9_Outcome                        = 0

        self.A_1_locker                         = 0
        self.B_2_locker                         = 0
        self.A_3_locker                         = 0
        self.B_4_locker                         = 0
        self.A_5_locker                         = 0
        self.B_6_locker                         = 0
        self.A_7_locker                         = 0
        self.B_8_locker                         = 0
        self.A_9_locker                         = 0

        self.delay                              = 0

        self.Player_A_goal = np.array([1, 0])
        self.Player_B_goal = np.array([0, 1])

    def start(self):

        if self.A_1 == 1:
            self.layer_input_A_1_inner                         = (np.random.random((self.layer_0_dim)) - 0.5) * 0
        else:
            self.layer_input_A_1_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_A_1_inner[self.A_1_Outcome]       = 10


        if self.B_2 ==1 :
            self.layer_input_B_2_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0
        else:
            self.layer_input_B_2_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_B_2_inner[self.B_2_Outcome]       = 10


        if self.A_3 ==1:
            self.layer_input_A_3_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0
        else:
            self.layer_input_A_3_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_A_3_inner[self.A_3_Outcome]       = 10


        if self.B_4 == 1:
            self.layer_input_B_4_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0
        else:
            self.layer_input_B_4_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_B_4_inner[self.B_4_Outcome]       = 10



        if self.A_5 == 1:
            self.layer_input_A_5_inner                         = (np.random.random((self.layer_0_dim)) - 0.5) * 0
        else:
            self.layer_input_A_5_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_A_5_inner[self.A_5_Outcome]       = 10


        if self.B_6 ==1 :
            self.layer_input_B_6_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0
        else:
            self.layer_input_B_6_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_B_6_inner[self.B_6_Outcome]       = 10


        if self.A_7 == 1:
            self.layer_input_A_7_inner                         = (np.random.random((self.layer_0_dim)) - 0.5) * 0
        else:
            self.layer_input_A_7_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_A_7_inner[self.A_7_Outcome]       = 10


        if self.B_8 ==1 :
            self.layer_input_B_8_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0
        else:
            self.layer_input_B_8_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_B_8_inner[self.B_8_Outcome]       = 10



        if self.A_9 == 1:
            self.layer_input_A_9_inner                         = (np.random.random((self.layer_0_dim)) - 0.5) * 0
        else:
            self.layer_input_A_9_inner                         = (np.random.random((self.layer_0_dim)) -0.5 ) * 0 - 10
            self.layer_input_A_9_inner[self.A_9_Outcome]       = 10



    def initialize_weights(self):

        synapse_layer_0_to_layer_1                                              = (np.random.random((self.layer_0_dim                                 , self.layer_1_dim                           )) -0.5 ) * 1
        synapse_layer_1_to_layer_2                                              = (np.random.random((self.layer_1_dim                                 , self.layer_2_dim                           )) -0.5 ) * 1
        synapse_layer_1_to_layer_write_1                                        = (np.random.random((self.layer_1_dim                                 , self.layer_write_1_dim                     )) -0.5 ) * 1
        synapse_layer_write_1_to_layer_memory_end_1                             = (np.random.random((self.layer_write_1_dim                           , self.layer_memory_end_1_dim                )) -0.5 ) * 1
        synapse_layer_memory_start_1_to_layer_memory_end_1                      =  np.array(np.ones(self.layer_memory_start_1_dim))
        synapse_layer_memory_end_1_to_layer_memory_start_1                      =  np.array(np.ones(self.layer_memory_start_1_dim))
        synapse_layer_1_to_future_layer_read_1                                  = (np.random.random((self.layer_1_dim                                 , self.layer_read_1_dim                      )) -0.5 ) * 1
        synapse_layer_1_to_future_layer_1                                       = (np.random.random((self.layer_1_dim                                 , self.layer_1_dim                           )) -0.5 ) * 1
        layer_write_1_size_controller                                           =  np.array(np.zeros(self.layer_write_1_dim))
        layer_read_1_enhancer                                                   =  np.array(np.ones(self.layer_read_1_dim)) * 3
        layer_writer_1_enhancer                                                 =  np.array(np.ones(self.layer_memory_end_1_dim) * 1)
        tanh_enhancer                                                           =  np.array(np.ones(self.layer_memory_end_1_dim) * 3.5)

        return  synapse_layer_0_to_layer_1,\
                synapse_layer_1_to_layer_2, \
                synapse_layer_1_to_layer_write_1, \
                synapse_layer_write_1_to_layer_memory_end_1, \
                synapse_layer_memory_start_1_to_layer_memory_end_1,\
                synapse_layer_memory_end_1_to_layer_memory_start_1, \
                synapse_layer_1_to_future_layer_1, \
                synapse_layer_1_to_future_layer_read_1, \
                layer_write_1_size_controller,\
                layer_read_1_enhancer,\
                layer_writer_1_enhancer,\
                tanh_enhancer

    def sigmoid(self, x):
        output = 1/(1+np.exp(-x))
        return output

    def sigmoid_output_to_derivative(self, output):
        return output*(1-output)

    def add_bias(self, X):
        X_new = np.ones(X.shape[0] + 1)
        X_new[1:] = X
        return X_new

    def tanh(self, x):
        output = np.tanh(x)
        return output

    def tanh_output_to_derivative(self, output):
        output = 1 - output*output
        return output

    def tanh_2(self, x):
        output = np.tanh(  3.5  *x)
        return output

    def tanh_2_output_to_derivative(self, output):
        output = 1 - output*output
        return output *     3.5

    def gate(self, x):
        output = 1/(1+np.exp(  -x))
        return output

    def gate_output_to_derivative(self, output):
        return output*(1-output)

    def generate_values_for_each_layer(self, selected_sentence):
        layer_memory_start_1_values = list()
        layer_memory_start_1_values.append(np.zeros(self.layer_memory_start_1_dim))
        layer_memory_start_1_values.append(np.zeros(self.layer_memory_start_1_dim))
        layer_memory_end_1_values   = layer_memory_start_1_values
        layer_0_values = list()
        layer_0_values.append(np.zeros(self.layer_0_dim))
        layer_0_values.append(np.zeros(self.layer_0_dim))
        layer_1_values = list()
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_1_values.append(np.zeros(self.layer_1_dim))
        layer_2_values = list()
        layer_2_values.append(np.zeros(self.layer_2_dim))
        layer_2_values.append(np.zeros(self.layer_2_dim))
        layer_write_1_values = list()
        layer_write_1_values.append(np.zeros(self.layer_write_1_dim))
        layer_write_1_values.append(np.zeros(self.layer_write_1_dim))
        layer_read_1_values = list()
        layer_read_1_values.append(np.zeros(self.layer_read_1_dim))
        layer_read_1_values.append(np.zeros(self.layer_read_1_dim))

        for position in range(np.array(selected_sentence).shape[0]):

            layer_0                   = selected_sentence[position]

            layer_memory_start_1      = layer_memory_end_1_values[-1] * self.synapse_layer_memory_end_1_to_layer_memory_start_1

            layer_read_1              = self.sigmoid(np.dot(layer_1_values[-1]               , self.synapse_layer_1_to_future_layer_read_1                                              ) )

            layer_1                   = self.sigmoid(np.dot(layer_0                          , self.synapse_layer_0_to_layer_1                                                          ) +
                                                     layer_memory_start_1 * layer_read_1 * self.layer_read_1_enhancer  +\
                                                     np.dot(layer_1_values[-1]               , self.synapse_layer_1_to_future_layer_1                                                   ) )

            layer_2                   = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_2                                                          ) )

            layer_write_1             = self.sigmoid(np.dot(layer_1                          , self.synapse_layer_1_to_layer_write_1                                                    ) )

            layer_write_1_sized       = layer_write_1 * self.sigmoid(self.layer_write_1_size_controller)

            layer_memory_end_1        = self.tanh(np.dot(layer_write_1_sized                 , self.synapse_layer_write_1_to_layer_memory_end_1                                         ) ) * self.layer_writer_1_enhancer +\
                                        ( layer_memory_start_1                               * self.synapse_layer_memory_start_1_to_layer_memory_end_1                                    )

            layer_memory_end_1        = self.tanh(layer_memory_end_1 * self.tanh_enhancer )


            layer_memory_start_1_values      .append(copy.deepcopy(    layer_memory_start_1         ))
            layer_memory_end_1_values        .append(copy.deepcopy(    layer_memory_end_1           ))
            layer_0_values                   .append(copy.deepcopy(    layer_0                      ))
            layer_1_values                   .append(copy.deepcopy(    layer_1                      ))
            layer_2_values                   .append(copy.deepcopy(    layer_2                      ))
            layer_write_1_values             .append(copy.deepcopy(    layer_write_1                ))
            layer_read_1_values              .append(copy.deepcopy(    layer_read_1                 ))

        return   layer_memory_start_1_values,\
                    layer_memory_end_1_values  ,\
                    layer_0_values             ,\
                    layer_1_values             ,\
                    layer_2_values             ,\
                    layer_write_1_values       ,\
                    layer_read_1_values

    def matrixation_delta(self, delta, dim_1, dim_2):
        prototype =  np.ones((dim_1, dim_2))
        for j in range(dim_2):
            for i in range(dim_1):
                prototype[i][j] = delta[0][j]
        return prototype

    def train_for_each(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1]])

        for position in range(np.array(selected_sentence).shape[0]):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1]])
            layer_0                                 = np.array([layer_0_values[-position - 1]])
            layer_1                                 = np.array([layer_1_values[-position - 1]])
            layer_2                                 = np.array([layer_2_values[-position - 1]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])



            self.synapse_layer_0_to_layer_1_update                        += np.atleast_2d(layer_0                                                                            ).T.dot(layer_1_delta                              )
            self.synapse_layer_1_to_layer_2_update                        += np.atleast_2d(layer_1                                                                            ).T.dot(layer_2_delta                              )
            self.synapse_layer_1_to_layer_write_1_update                  += np.atleast_2d(layer_1                                                                            ).T.dot(layer_write_1_delta                        )
            self.synapse_layer_write_1_to_layer_memory_end_1_update       += np.atleast_2d(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller]))       ).T.dot(layer_memory_end_1_delta_2                 )
            self.synapse_layer_1_to_future_layer_read_1_update            += np.atleast_2d(layer_1                                                                            ).T.dot(future_layer_read_1_delta                  )
            self.synapse_layer_1_to_future_layer_1_update                 += np.atleast_2d(layer_1                                                                            ).T.dot(future_layer_1_delta                       )
            self.layer_write_1_size_controller_update                     += layer_write_1_sized_delta[0]
            self.layer_read_1_enhancer_update                             += layer_read_1_enhancer_delta[0]
            self.layer_writer_1_enhancer_update                           += layer_write_1_enhancer_delta[0]
            self.tanh_enhancer_update                                     += tanh_enhancer_delta[0]

            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.synapse_layer_0_to_layer_1                                   += self.synapse_layer_0_to_layer_1_update                   * self.alpha
        self.synapse_layer_1_to_layer_2                                   += self.synapse_layer_1_to_layer_2_update                   * self.alpha
        self.synapse_layer_1_to_layer_write_1                             += self.synapse_layer_1_to_layer_write_1_update             * self.alpha
        self.synapse_layer_write_1_to_layer_memory_end_1                  += self.synapse_layer_write_1_to_layer_memory_end_1_update  * self.alpha
        self.synapse_layer_1_to_future_layer_read_1                       += self.synapse_layer_1_to_future_layer_read_1_update       * self.alpha
        self.synapse_layer_1_to_future_layer_1                            += self.synapse_layer_1_to_future_layer_1_update            * self.alpha
        self.layer_write_1_size_controller                                += self.layer_write_1_size_controller_update                * self.alpha
        self.layer_read_1_enhancer                                        += self.layer_read_1_enhancer_update                        * self.alpha
        self.layer_writer_1_enhancer                                      += self.layer_writer_1_enhancer_update                      * self.alpha * 0.05
        self.tanh_enhancer                                                += self.tanh_enhancer_update                                * self.alpha

        self.synapse_layer_0_to_layer_1_update                            *= 0
        self.synapse_layer_1_to_layer_2_update                            *= 0
        self.synapse_layer_1_to_layer_write_1_update                      *= 0
        self.synapse_layer_write_1_to_layer_memory_end_1_update           *= 0
        self.synapse_layer_1_to_future_layer_read_1_update                *= 0
        self.synapse_layer_1_to_future_layer_1_update                     *= 0
        self.layer_write_1_size_controller_update                         *= 0
        self.layer_read_1_enhancer_update                                 *= 0
        self.layer_writer_1_enhancer_update                               *= 0
        self.tanh_enhancer_update                                         *= 0





    def fit(self, X, Y):
        for j in range(self.epochs):

            random_int        = np.random.randint(X.shape[0])
            selected_sentence = np.array( X[random_int ] )
            selected_result   = np.array( Y[random_int ] )

            layer_memory_start_1_values,\
            layer_memory_end_1_values,\
            layer_0_values,\
            layer_1_values, \
            layer_2_values, \
            layer_write_1_values, \
            layer_read_1_values  = self.generate_values_for_each_layer(selected_sentence)

            self.train_for_each(selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       selected_result)

        return self






#-------------------------------------------------------------------------------------------------




    def train_for_input_A_1(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_A_1_inner += layer_0_delta_list[-1] * self.beta



    def train_for_input_B_2(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_B_2_inner += layer_0_delta_list[-2] * self.beta

    def train_for_input_A_3(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_A_3_inner += layer_0_delta_list[-3] * self.beta


    def train_for_input_B_4(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_B_4_inner += layer_0_delta_list[-4] * self.beta


    def train_for_input_A_5(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0] - self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_A_5_inner += layer_0_delta_list[-5] * self.beta


    def train_for_input_B_6(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]-  self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_B_6_inner += layer_0_delta_list[-6 ] * self.beta

    def train_for_input_A_7(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_A_7_inner += layer_0_delta_list[-7] * self.beta

    def train_for_input_B_8(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_B_8_inner += layer_0_delta_list[-8] * self.beta


    def train_for_input_A_9(self, selected_sentence,
                       layer_memory_start_1_values,
                       layer_memory_end_1_values,
                       layer_0_values,
                       layer_1_values,
                       layer_2_values,
                       layer_write_1_values,
                       layer_read_1_values,
                       layer_2_opposite_value):

        future_layer_memory_start_1_delta           = np.array([np.zeros(self.layer_memory_start_1_dim)])
        future_layer_1_delta                        = np.array([np.zeros(self.layer_1_dim)])
        future_layer_read_1_delta                   = np.array([np.zeros(self.layer_read_1_dim)])
        layer_2_error                               = np.array([layer_2_opposite_value - layer_2_values[-1- self.delay]])

        layer_0_delta_list = list()

        for position in range(np.array(selected_sentence).shape[0]- self.delay):

            layer_memory_start_1                    = np.array([layer_memory_start_1_values[-position - 1- self.delay]])
            layer_memory_end_1                      = np.array([layer_memory_end_1_values[-position - 1- self.delay]])
            layer_0                                 = np.array([layer_0_values[-position - 1- self.delay]])
            layer_1                                 = np.array([layer_1_values[-position - 1- self.delay]])
            layer_2                                 = np.array([layer_2_values[-position - 1- self.delay]])
            layer_write_1                           = np.array([layer_write_1_values[-position - 1- self.delay]])
            layer_read_1                            = np.array([layer_read_1_values[-position - 1- self.delay]])


            layer_memory_end_1_delta_1              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * 1

            layer_memory_end_1_delta_2              = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * np.array([self.layer_writer_1_enhancer])

            layer_write_1_enhancer_delta            = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * self.tanh_enhancer * self.tanh_output_to_derivative(self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) ) * self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1))

            tanh_enhancer_delta                     = future_layer_memory_start_1_delta                                                                           * self.tanh_output_to_derivative(layer_memory_end_1) * (self.tanh(np.dot(layer_write_1 * self.sigmoid(np.array([self.layer_write_1_size_controller])), self.synapse_layer_write_1_to_layer_memory_end_1)) * np.array([self.layer_writer_1_enhancer]) + ( layer_memory_start_1 * self.synapse_layer_memory_start_1_to_layer_memory_end_1) )

            layer_write_1_delta                     = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * self.sigmoid_output_to_derivative(layer_write_1) * self.sigmoid(np.array([self.layer_write_1_size_controller]))

            layer_write_1_sized_delta               = (layer_memory_end_1_delta_2.dot(self.synapse_layer_write_1_to_layer_memory_end_1.T                      ) ) * layer_write_1                                    * self.sigmoid_output_to_derivative(self.sigmoid(np.array([self.layer_write_1_size_controller])))

            layer_2_delta                           = (layer_2_error                                                                                            ) * self.sigmoid_output_to_derivative(layer_2)

            layer_1_delta                           = (layer_2_delta.dot(self.synapse_layer_1_to_layer_2.T                                                    ) +
                                                       future_layer_read_1_delta.dot(self.synapse_layer_1_to_future_layer_read_1.T                            ) +
                                                       future_layer_1_delta.dot(self.synapse_layer_1_to_future_layer_1.T                                      ) +
                                                       layer_write_1_delta.dot(self.synapse_layer_1_to_layer_write_1.T                                        ) ) * self.sigmoid_output_to_derivative(layer_1)

            layer_read_1_delta                      = layer_1_delta  * layer_memory_start_1 * self.sigmoid_output_to_derivative(layer_read_1) * np.array([self.layer_read_1_enhancer])

            layer_read_1_enhancer_delta             = layer_1_delta  * layer_memory_start_1 * layer_read_1                                    * 1

            layer_memory_start_1_delta              = layer_memory_end_1_delta_1 +\
                                                      layer_1_delta  * 1                    * layer_read_1                                    * np.array([ self.layer_read_1_enhancer])

            layer_0_delta                           = (layer_1_delta.dot(self.synapse_layer_0_to_layer_1.T                                                    ) ) * self.sigmoid_output_to_derivative(layer_0)


            layer_0_delta_list.append(layer_0_delta[0])


            future_layer_memory_start_1_delta       = layer_memory_start_1_delta
            future_layer_1_delta                    = layer_1_delta
            future_layer_read_1_delta               = layer_read_1_delta
            layer_2_error                           = np.array([np.zeros(self.layer_2_dim)])

        self.layer_input_A_9_inner += layer_0_delta_list[-9] * self.beta


    def predict(self):




        for j in range(self.epochs):



            #----------------------------------------------------------------------------------------------------

            if self.A_9 == 1  & self.A_9_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)


                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ])  )

                desired_result = self.Player_A_goal
                # desired_result[1] = layer_2_values[-1][1]  #<----------------------------------------------------------

                self.train_for_input_A_9(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                           layer_memory_start_1_values,
                           layer_memory_end_1_values,
                           layer_0_values,
                           layer_1_values,
                           layer_2_values,
                           layer_write_1_values,
                           layer_read_1_values,
                           desired_result)

            #----------------------------------------------------------------------------------------------------

            if self.B_8 == 1 & self.B_8_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)

                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) )

                desired_result = self.Player_B_goal
                # desired_result[0] = layer_2_values[-1][0]      #<----------------------------------------------------------

                self.train_for_input_B_8( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                                     layer_memory_start_1_values,
                                     layer_memory_end_1_values,
                                     layer_0_values,
                                     layer_1_values,
                                     layer_2_values,
                                     layer_write_1_values,
                                     layer_read_1_values,
                                     desired_result)


            #----------------------------------------------------------------------------------------------------

            if self.A_7 == 1 & self.A_7_locker == 1:
                for i in range(10):
                    layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                    layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                    layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                    layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                    layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                    layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                    layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                    layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                    layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)


                    layer_memory_start_1_values,\
                    layer_memory_end_1_values,\
                    layer_0_values,\
                    layer_1_values, \
                    layer_2_values, \
                    layer_write_1_values, \
                    layer_read_1_values  = self.generate_values_for_each_layer(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ])  )

                    desired_result = self.Player_A_goal
                    # desired_result[1] = layer_2_values[-1][1]  #<----------------------------------------------------------

                    self.train_for_input_A_7(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                               layer_memory_start_1_values,
                               layer_memory_end_1_values,
                               layer_0_values,
                               layer_1_values,
                               layer_2_values,
                               layer_write_1_values,
                               layer_read_1_values,
                               desired_result)

            #----------------------------------------------------------------------------------------------------

            if self.B_6 == 1 & self.B_6_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)

                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) )

                desired_result = self.Player_B_goal
                # desired_result[0] = layer_2_values[-1][0]      #<----------------------------------------------------------

                self.train_for_input_B_6( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                                     layer_memory_start_1_values,
                                     layer_memory_end_1_values,
                                     layer_0_values,
                                     layer_1_values,
                                     layer_2_values,
                                     layer_write_1_values,
                                     layer_read_1_values,
                                     desired_result)

            #----------------------------------------------------------------------------------------------------

            if self.A_5 == 1 & self.A_5_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)


                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ])  )

                desired_result = self.Player_A_goal
                # desired_result[1] = layer_2_values[-1][1]  #<----------------------------------------------------------

                self.train_for_input_A_5(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                           layer_memory_start_1_values,
                           layer_memory_end_1_values,
                           layer_0_values,
                           layer_1_values,
                           layer_2_values,
                           layer_write_1_values,
                           layer_read_1_values,
                           desired_result)

            #----------------------------------------------------------------------------------------------------

            if self.B_4 == 1 & self.B_4_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)

                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) )

                desired_result = self.Player_B_goal
                # # desired_result[0] = layer_2_values[-1][0]      #<----------------------------------------------------------

                self.train_for_input_B_4( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                                     layer_memory_start_1_values,
                                     layer_memory_end_1_values,
                                     layer_0_values,
                                     layer_1_values,
                                     layer_2_values,
                                     layer_write_1_values,
                                     layer_read_1_values,
                                     desired_result)


            #----------------------------------------------------------------------------------------------------

            if self.A_3 == 1 & self.A_3_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)


                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ])  )

                desired_result = self.Player_A_goal
                # # desired_result[1] = layer_2_values[-1][1]  #<----------------------------------------------------------

                self.train_for_input_A_3(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                           layer_memory_start_1_values,
                           layer_memory_end_1_values,
                           layer_0_values,
                           layer_1_values,
                           layer_2_values,
                           layer_write_1_values,
                           layer_read_1_values,
                           desired_result)


            #----------------------------------------------------------------------------------------------------

            if self.B_2 == 1 & self.B_2_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)

                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) )

                desired_result = self.Player_B_goal
                # # desired_result[0] = layer_2_values[-1][0]      #<----------------------------------------------------------

                self.train_for_input_B_2( np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                                     layer_memory_start_1_values,
                                     layer_memory_end_1_values,
                                     layer_0_values,
                                     layer_1_values,
                                     layer_2_values,
                                     layer_write_1_values,
                                     layer_read_1_values,
                                     desired_result)

            #----------------------------------------------------------------------------------------------------

            if self.A_1 == 1 & self.A_1_locker == 1:

                layer_input_A_1 = self.sigmoid(self.layer_input_A_1_inner)
                layer_input_B_2 = self.sigmoid(self.layer_input_B_2_inner)
                layer_input_A_3 = self.sigmoid(self.layer_input_A_3_inner)
                layer_input_B_4 = self.sigmoid(self.layer_input_B_4_inner)
                layer_input_A_5 = self.sigmoid(self.layer_input_A_5_inner)
                layer_input_B_6 = self.sigmoid(self.layer_input_B_6_inner)
                layer_input_A_7 = self.sigmoid(self.layer_input_A_7_inner)
                layer_input_B_8 = self.sigmoid(self.layer_input_B_8_inner)
                layer_input_A_9 = self.sigmoid(self.layer_input_A_9_inner)

                layer_memory_start_1_values,\
                layer_memory_end_1_values,\
                layer_0_values,\
                layer_1_values, \
                layer_2_values, \
                layer_write_1_values, \
                layer_read_1_values  = self.generate_values_for_each_layer(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ])  )

                desired_result = self.Player_A_goal
                # desired_result[1] = layer_2_values[-1][1]  #<----------------------------------------------------------

                self.train_for_input_A_1(  np.array([ layer_input_A_1, layer_input_B_2, layer_input_A_3, layer_input_B_4, layer_input_A_5, layer_input_B_6, layer_input_A_7, layer_input_B_8, layer_input_A_9  ]) ,
                           layer_memory_start_1_values,
                           layer_memory_end_1_values,
                           layer_0_values,
                           layer_1_values,
                           layer_2_values,
                           layer_write_1_values,
                           layer_read_1_values,
                           desired_result)


        return self




    def predict_2(self, selected_sentence):

        layer_memory_start_1_values, \
        layer_memory_end_1_values, \
        layer_0_values, \
        layer_1_values, \
        layer_2_values, \
        layer_write_1_values, \
        layer_read_1_values = self.generate_values_for_each_layer(selected_sentence)

        return np.array(layer_2_values[-1])

    def count_and_print_success_rate(self, X, Y):

        self.number_of_correctness = 0

        for i in range(X.shape[0]):

            if np.argmax( self.predict_2(X[i]) ) == np.argmax( np.array( Y[i] )):
                self.number_of_correctness += 1

            if np.argmax( self.predict_2(X[i]) ) != np.argmax( np.array( Y[i] )):
                print('The failed target is line ', i+1)

        print(' Success rate = ', (self.number_of_correctness/X.shape[0]) * 100)
        print(' The model used is Recurrent_Hippocampus')
        print(' The epochs is', self.epochs)
