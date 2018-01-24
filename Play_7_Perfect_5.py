

import numpy as np
import math




A_1              = 0                                                                                                                 ######<<---------------------------------
B_2              = 0                                                                                                                 ######<<---------------------------------
A_3              = 0                                                                                                                 ######<<---------------------------------
B_4              = 0                                                                                                                 ######<<---------------------------------
A_5              = 0                                                                                                                 ######<<---------------------------------
B_6              = 0                                                                                                                 ######<<---------------------------------
A_7              = 0                                                                                                                 ######<<---------------------------------
B_8              = 1                                                                                                                 ######<<---------------------------------
A_9              = 1                                                                                                                 ######<<---------------------------------

layer            = 7                                                                                                                            ######<<---------------------------------



A_1_locker       = 0                                                                                                                 ######<<---------------------------------
B_2_locker       = 0                                                                                                                 ######<<---------------------------------
A_3_locker       = 0                                                                                                                 ######<<---------------------------------
B_4_locker       = 0                                                                                                                 ######<<---------------------------------
A_5_locker       = 0                                                                                                                 ######<<---------------------------------
B_6_locker       = 0                                                                                                                 ######<<---------------------------------
A_7_locker       = 0                                                                                                                 ######<<---------------------------------
B_8_locker       = 1                                                                                                                 ######<<---------------------------------
A_9_locker       = 1                                                                                                                 ######<<---------------------------------

delay            = 0                                                                                                                         ######<<---------------------------------

A_1_locker_2     = 0                                                                                                                 ######<<---------------------------------
B_2_locker_2     = 0                                                                                                                 ######<<---------------------------------
A_3_locker_2     = 0                                                                                                                 ######<<---------------------------------
B_4_locker_2     = 0                                                                                                                 ######<<---------------------------------
A_5_locker_2     = 0                                                                                                                 ######<<---------------------------------
B_6_locker_2     = 0                                                                                                                 ######<<---------------------------------
A_7_locker_2     = 0                                                                                                                 ######<<---------------------------------
B_8_locker_2     = 1                                                                                                                 ######<<---------------------------------
A_9_locker_2     = 0                                                                                                                 ######<<---------------------------------

delay_2          = 1                                                                                                                         ######<<---------------------------------

A_1_Outcome      = 2   # 4   # 4     # 4      # 4         # 4      # 4       # 4        # 4        # 4       # 4      #  0      #  4     #  2    #  2     #  2                                                                                     ######<<---------------------------------
B_2_Outcome      = 3   # 8   # 8     # 0      # 0         # 2      # 2       # 6        # 6        # 8       # 8      #  8      #  3     #  3    #  3     #  3                                                                                    ######<<---------------------------------
A_3_Outcome      = 7   # 2   # 6     # 2      # 6         # 0      # 8       # 0        # 8        # 3       # 5      #  5      #  5     #  5    #  5                                                                                         ######<<---------------------------------
B_4_Outcome      = 4   # 6   # 2     # 6      # 2         # 8      # 0       # 8        # 0        # 5       # 3      #  3      #  6     #  6    #                                                                                         ######<<---------------------------------
A_5_Outcome      = 5   # 7   # 5     # 3      # 1         # 5      # 1       # 7        # 3        # 2       # 2      #  2      #  8     #                                                                                            ######<<---------------------------------
B_6_Outcome      = 8   # 1   # 3     # 5      # 7         # 3      # 7       # 1        # 5        # 6       # 6      #  6      #                                                                                                   ######<<---------------------------------
A_7_Outcome      = 0   # 5   # 7     # 1      # 3         # 1      # 5       # 3        # 2        # 7       # 7      #                                                                                                          ######<<---------------------------------參考
B_8_Outcome      = 1   # 3   # 1     # 7      # ?         # 7      # 3       # 5        # ?         # 1       # ?                                                                                                                ######<<---------------------------------參考
A_9_Outcome      = 1   # 0   # 0     # 8      # ?         # 6      # 6       # 2        # ?         # 0       # ?                                                                                                                     ######<<---------------------------------參考
                                                                           # 參考 作為確認是否決定要讓自己當下獲勝之用  奇數預測型態是找出對自己有利的 偶數預測型態是找出對自己不利的


          # B_4                # 6   # 2      #6       # 2         # 8      # 0      # 8        #0         # 5         size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      layered  ******** \O_O/    \O_O/    \O_O/   \O_O/  \ O_O/!!!!


          # A_5                # 7   # 5      #3       # 1         # 5      # 1      # 7        #3         # 2         size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      layered  ******** \O_O/    \O_O/    \O_O/   \O_O/  \ O_O/!!!!


          # B_6                # 1   # 3      #5       # 7         # 3      # 7      # 1        #5         # 6         size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      layered  ******** \O_O/    \O_O/    \O_O/   \O_O/  \ O_O/!!!!
          # B_6                # 1   # 0      #7       # 7         # 3      # 7      # 1        #7         # 6         size+20     select = 200       alpha,beta = 0.5       epcohs =400         layered  *****       Q__Q

          # A_7                # 5   # 0      #8       # 3         # 6      # 6      # 5        #2         # 7         size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      layered  ******** \O_O/    \O_O/    \O_O/   \O_O/  \ O_O/!!!!


















          # A_3                # x   # x      #x       # x         # x      # x      # x        #x        size+20     select = 5000     alpha,beta = 0.5       epcohs = 8000      layered


          # A_5                # 7   # 7      #x       # 7         # x      # 7       # x        #x      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2 + 0.5*j, -1 -0.5*j]

          # A_5                # x   # 7      #x       # x         # x      # 7       # x        #x      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2 + 0.2.5*j, -1 -0.2.5*j]

          # A_5                # 7   # 7      #x       # 7         # 7      # x       # x        #5      size+20     select = 10000   alpha,beta = 0.5       epcohs = 10000    payoff = [2 + 0.2*j, -1 -0.2*j]      *
          # A_5                # 0   # 3      #x       # x         # 7      # x       # x        #x      size+20     select = 5000     alpha,beta = 0.5       epcohs = 8000      payoff = [2 + 0.2*j, -1 -0.2*j]

          # A_5                # x   # 5      #7       # x         # 3      # 7       # 7        #3      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [3 + 0.15*j, -2-0.15*j]      ***
          # A_5                # 5   # 7      #3       # 7         # 5      # 3       # 1        #3      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2 + 0.2*j, -1 -0.2*j]      ***
          # A_5                # x   # 7      #x       # 1         # 7      # 3       # x        #x      size+20     select = 2000     alpha,beta = 0.5       epcohs = 5000      payoff = [2 + 0.2*j, -1 -0.2*j]      *

          # A_5                # x   # 3      #x       # 7         # x      # 1       # x        #x      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2 + 0.1*j, -1 -0.1*j]

          # A_5                # x   # x      #x       # x         # x      # x       # x        #x      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2 - 0.2*j, -1 + 0.2*j]  j
          # A_5                # x   # x      #x       # 7         # 3      # 5       # 2        #3      size+20     select = 5000     alpha,beta = 0.5       epcohs = 8000      payoff = [2, -1]
          # A_5                # x   # x      #x       # x         # 7      # x       # x        #7      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [3, -2]
          # A_5                # x   # x      #x       # x         # 7      # 7       # 7        #7      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2, -1]     [2, -1]
          # A_5                # 5   # 0      #3       # 3         # 7      # 7       # 5        #3      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000      payoff = [2, -1]
          # A_5                # x   # x      #x       # x         # 1      # 7       # x        #x      size+20     select = 30000   alpha,beta = 0.5       epcohs = 90000
          # A_5                # 1   # 0      #5       # 7         # 3      # x       # x        #5      size+20     select = 20000   alpha,beta = 0.5       epcohs = 40000
          # A_5                # 1   # 7      #5       # 7         # 3      # x       # x        #x      size+20     select = 10000   alpha,beta = 0.5       epcohs = 20000
          # A_5                # 3   # 3      #x       # 5         # 1      # 7       # 1        #5      size+20     select = 7000     alpha,beta = 0.5       epcohs = 12000
          # A_5                # 0   # 3      #5       # 7         # 7      # 6       # 3        #1      size+20     select = 5000     alpha,beta = 0.5       epcohs = 8000
          # A_5                # 1   # 1      #5       # 5         # 3      # x       # x        #x      size+20     select = 4000     alpha,beta = 0.5       epcohs = 7000
          # A_5                # 1   # 1      #8       # 8         # 3      # 6       # 1        #5      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000
          # A_5                # x   # x      #x       # x         # x      # x       # x        #x      size+20     select = 1000     alpha,beta = 0.5       epcohs = 2000
          # B_6                # 1   # 3      #5       # 7         # 3      # 7       # 1        #5      size+20     select = 3500     alpha,beta = 0.5       epcohs = 6000  *******






# Strategies =          [[0, 0, 0, 0, 1, 0, 0, 0, 0]   , [0, 0, 1, 0, 0, 0, 0, 0, 0]     , [0, 0, 0, 0, 0, 0, 0, 0, 1]    , [1, 0, 0, 0, 0, 0, 0, 0, 0] ]#  ,    [0, 0, 0, 0, 0, 0, 0, 1, 0]      ,   [0, 1, 0, 0, 0, 0, 0, 0, 0]    ]
# Player_A_Strategies = [[0, 0, 0, 0, 1, 0, 0, 0, 0]  ,  [0, 0, 0, 0, 0, 0, 0, 0, 1]    ]# , [0, 0, 0, 0, 0, 0, 0, 1, 0]        ,[0, 0, 0, 1, 0, 0, 0, 0, 0]           ]
# Player_B_Strategies = [[0, 0, 1, 0, 0, 0, 0, 0, 0]      ,   [1, 0, 0, 0, 0, 0, 0, 0, 0]     ]#      , [0, 1, 0, 0, 0, 0, 0, 0, 0]                                                  ]

Success_list = list()

Ending_1 = [ [1, 1, 1],
             [0, 0, 0],
             [0, 0, 0]  ]
Success_list.append(Ending_1)

Ending_2 = [ [0, 0, 0],
             [1, 1, 1],
             [0, 0, 0]  ]
Success_list.append(Ending_2)

Ending_3 = [ [0, 0, 0],
             [0, 0, 0],
             [1, 1, 1]  ]
Success_list.append(Ending_3)

Ending_4 = [ [1, 0, 0],
             [1, 0, 0],
             [1, 0, 0]  ]
Success_list.append(Ending_4)

Ending_5 = [ [0, 1, 0],
             [0, 1, 0],
             [0, 1, 0]  ]
Success_list.append(Ending_5)

Ending_6 = [ [0, 0, 1],
             [0, 0, 1],
             [0, 0, 1]  ]
Success_list.append(Ending_6)

Ending_7 = [ [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]  ]
Success_list.append(Ending_7)


Ending_8 = [ [0, 0, 1],
             [0, 1, 0],
             [1, 0, 0]  ]
Success_list.append(Ending_8)



Ending_1 = np.reshape(Ending_1, 9)
Ending_2 = np.reshape(Ending_2, 9)
Ending_3 = np.reshape(Ending_3, 9)
Ending_4 = np.reshape(Ending_4, 9)
Ending_5 = np.reshape(Ending_5, 9)
Ending_6 = np.reshape(Ending_6, 9)
Ending_7 = np.reshape(Ending_7, 9)
Ending_8 = np.reshape(Ending_8, 9)



X = list()
Y = list()



layer += 1

def generate():
    for i in range(3500):      #5000   > 2000 > 3000 .....                                                                                                                            ######<<---------------------------------



        Strategies = list()
        Player_A_Strategies = list()
        Player_B_Strategies = list()

        if A_1 == 0:
            Player_A_Strategy = np.zeros(9)
            Player_A_Strategy[A_1_Outcome] = 1
            Strategies.append(Player_A_Strategy)
            Player_A_Strategies.append(Player_A_Strategy)

        if B_2 == 0:
            Player_B_Strategy = np.zeros(9)
            Player_B_Strategy[B_2_Outcome] = 1
            Strategies.append(Player_B_Strategy)
            Player_B_Strategies.append(Player_B_Strategy)

        if A_3 == 0:
            Player_A_Strategy = np.zeros(9)
            Player_A_Strategy[A_3_Outcome] = 1
            Strategies.append(Player_A_Strategy)
            Player_A_Strategies.append(Player_A_Strategy)

        if B_4 == 0:
            Player_B_Strategy = np.zeros(9)
            Player_B_Strategy[B_4_Outcome] = 1
            Strategies.append(Player_B_Strategy)
            Player_B_Strategies.append(Player_B_Strategy)

        if A_5 == 0:
            Player_A_Strategy = np.zeros(9)
            Player_A_Strategy[A_5_Outcome] = 1
            Strategies.append(Player_A_Strategy)
            Player_A_Strategies.append(Player_A_Strategy)

        if B_6 == 0:
            Player_B_Strategy = np.zeros(9)
            Player_B_Strategy[B_6_Outcome] = 1
            Strategies.append(Player_B_Strategy)
            Player_B_Strategies.append(Player_B_Strategy)

        if A_7 == 0:
            Player_A_Strategy = np.zeros(9)
            Player_A_Strategy[A_7_Outcome] = 1
            Strategies.append(Player_A_Strategy)
            Player_A_Strategies.append(Player_A_Strategy)

        if B_8 == 0:
            Player_B_Strategy = np.zeros(9)
            Player_B_Strategy[B_8_Outcome] = 1
            Strategies.append(Player_B_Strategy)
            Player_B_Strategies.append(Player_B_Strategy)

        if A_9 == 0:
            Player_A_Strategy = np.zeros(9)
            Player_A_Strategy[A_9_Outcome] = 1
            Strategies.append(Player_A_Strategy)
            Player_A_Strategies.append(Player_A_Strategy)

        Outcome = [0, 0]

        for j in range(  math.ceil(  (9 - layer + 1 )/ 2 )  ):

            if (Outcome == [1, 0]) | (Outcome ==[0, 1])| (Outcome ==[-1, 2])| (Outcome ==[2, -1])| (Outcome ==[-2, 3])| (Outcome ==[3, -2]) |  (  Outcome == [2 + 0.2 * (j-1), -1 - 0.2 * (j-1)]  ) | (  Outcome == [-1 - 0.2 * (j-1), 2 + 0.2 * (j-1)]  )  :
                break

            else:


                Player_A_index = np.random.randint(9)
                Player_A_Strategy = np.zeros(9)
                Player_A_Strategy[Player_A_index] = 1
                if (sum( np.array(Strategies) ) * Player_A_Strategy != [0, 0, 0, 0, 0, 0, 0, 0, 0]).any():
                    Outcome = [0, 1]

                Player_A_Strategies.append(Player_A_Strategy)
                Strategies.append(Player_A_Strategy)

                if (Outcome != [0, 1]) & (Outcome != [-1, 2]) & (Outcome != [-2, 3]) :
                    for k in range(8):
                        if ( sum( np.array(Player_A_Strategies) )  * np.reshape(Success_list[k], 9 ) == np.reshape(Success_list[k], 9 ) ).all():
                            Outcome = [1, 0]# [2 + 0.2 * j, -1 - 0.2 * j]




                if (Outcome != [1, 0]) & (Outcome != [0, 1]) & (Outcome != [-1, 2]) & (Outcome != [2, - 1]) & (Outcome != [-2, 3]) & (Outcome != [3, - 2]) &   ( Outcome != [2 + 0.2 * j, -1 - 0.2 * j])          &   (j !=     math.ceil((9 - layer + 1 )/ 2 + 0.5 ) - 1     ):                                                                       ######<<---------------------------------


                    Player_B_index = np.random.randint(9)
                    Player_B_Strategy = np.zeros(9)
                    Player_B_Strategy[Player_B_index] = 1
                    if (sum( np.array(Strategies) ) * Player_B_Strategy != [0, 0, 0, 0, 0, 0, 0, 0, 0]).any():
                        Outcome = [1, 0]

                    Player_B_Strategies.append(Player_B_Strategy)
                    Strategies.append(Player_B_Strategy)

                    if (Outcome != [1, 0]) & (Outcome != [2, -1])& (Outcome != [3, -2]):
                        for k in range(8):
                            if ( sum( np.array(Player_B_Strategies) )  * np.reshape(Success_list[k], 9 ) == np.reshape(Success_list[k], 9 ) ).all():
                                Outcome = [0,1] # [ -1 - 0.2 * j, 2 + 0.2 * j]



        X.append(Strategies)
        Y.append(Outcome)


generate()

X = np.array(X)
Y = np.array(Y)

#=================================USING THE MODEL===============================================================

from Game_Net_7_5 import *   # 9-7 i orthadal

alpha          = 0.5        ## 0.5                                                    ######<<---------------------------------
beta           = 10            #                                                                                       ######<<---------------------------------
epochs         = 6000        # 10000 > ......            # <<=====================                             ######<<---------------------------------



Game           = Game_Net( 9  , alpha, beta, epochs)  # <<=====================

Game.A_1                                = A_1
Game.B_2                                = B_2
Game.A_3                                = A_3
Game.B_4                                = B_4
Game.A_5                                = A_5
Game.B_6                                = B_6
Game.A_7                                = A_7
Game.B_8                                = B_8
Game.A_9                                = A_9
Game.A_1_Outcome                        = A_1_Outcome
Game.B_2_Outcome                        = B_2_Outcome
Game.A_3_Outcome                        = A_3_Outcome
Game.B_4_Outcome                        = B_4_Outcome
Game.A_5_Outcome                        = A_5_Outcome
Game.B_6_Outcome                        = B_6_Outcome
Game.A_7_Outcome                        = A_7_Outcome
Game.B_8_Outcome                        = B_8_Outcome
Game.A_9_Outcome                        = A_9_Outcome


Game.A_1_locker                         = A_1_locker
Game.B_2_locker                         = B_2_locker
Game.A_3_locker                         = A_3_locker
Game.B_4_locker                         = B_4_locker
Game.A_5_locker                         = A_5_locker
Game.B_6_locker                         = B_6_locker
Game.A_7_locker                         = A_7_locker
Game.B_8_locker                         = B_8_locker
Game.A_9_locker                         = A_9_locker
Game.delay                              = delay

Game.start()

Game.fit(X, Y)

Game.predict()

#><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


alpha          = 0.5        ## 0.5                                                    ######<<---------------------------------
beta           = 0.5            #                                                                                       ######<<---------------------------------
epochs         = 6000        # 10000 > ......            # <<=====================                             ######<<---------------------------------

Game_2         = Game_Net( 9  , alpha, beta, epochs)  # <<=====================

Game_2.A_1                                = A_1
Game_2.B_2                                = B_2
Game_2.A_3                                = A_3
Game_2.B_4                                = B_4
Game_2.A_5                                = A_5
Game_2.B_6                                = B_6
Game_2.A_7                                = A_7
Game_2.B_8                                = B_8
Game_2.A_9                                = A_9
Game_2.A_1_Outcome                        = A_1_Outcome
Game_2.B_2_Outcome                        = B_2_Outcome
Game_2.A_3_Outcome                        = A_3_Outcome
Game_2.B_4_Outcome                        = B_4_Outcome
Game_2.A_5_Outcome                        = A_5_Outcome
Game_2.B_6_Outcome                        = B_6_Outcome
Game_2.A_7_Outcome                        = A_7_Outcome
Game_2.B_8_Outcome                        = B_8_Outcome
Game_2.A_9_Outcome                        = A_9_Outcome


Game_2.A_1_locker                         = A_1_locker_2
Game_2.B_2_locker                         = B_2_locker_2
Game_2.A_3_locker                         = A_3_locker_2
Game_2.B_4_locker                         = B_4_locker_2
Game_2.A_5_locker                         = A_5_locker_2
Game_2.B_6_locker                         = B_6_locker_2
Game_2.A_7_locker                         = A_7_locker_2
Game_2.B_8_locker                         = B_8_locker_2
Game_2.A_9_locker                         = A_9_locker_2
Game_2.delay                              = delay_2


Game_2.start()

Game_2.fit(X, Y)

Game_2.predict()


#><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


print('Final outcome')




print('Player_A_1_Strategy')
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_A_1_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_A_1_inner), (3, 3)))


print('Player_B_2_Strategy')
Game.layer_input_B_2_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_B_2_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_B_2_inner), (3, 3)))


print('Player_A_3_Strategy')
Game.layer_input_A_3_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_A_3_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_A_3_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_A_3_inner), (3, 3)))


print('Player_B_4_first_Strategy')
Game_2.layer_input_B_4_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game_2.layer_input_B_4_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game_2.layer_input_B_4_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game_2.sigmoid(Game_2.layer_input_B_4_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game_2.sigmoid(Game_2.layer_input_B_4_inner), (3, 3)))

print('Player_B_4_second_Strategy')
Game.layer_input_B_4_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_B_4_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_B_4_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_B_4_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_B_4_inner), (3, 3)))

print('Player_A_5_first_Strategy')
Game_2.layer_input_A_5_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game_2.layer_input_A_5_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game_2.layer_input_A_5_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game_2.layer_input_A_5_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game_2.sigmoid(Game_2.layer_input_A_5_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game_2.sigmoid(Game_2.layer_input_A_5_inner), (3, 3)))

print('Player_A_5_second_Strategy')
Game.layer_input_A_5_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_A_5_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_A_5_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game.layer_input_A_5_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_A_5_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_A_5_inner), (3, 3)))


print('Player_B_6_first_Strategy')
Game_2.layer_input_B_6_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game_2.layer_input_B_6_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game_2.layer_input_B_6_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game_2.layer_input_B_6_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game_2.layer_input_B_6_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game_2.sigmoid(Game_2.layer_input_B_6_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game_2.sigmoid(Game_2.layer_input_B_6_inner), (3, 3)))


print('Player_B_6_second_Strategy')
Game.layer_input_B_6_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_B_6_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_B_6_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game.layer_input_B_6_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game.layer_input_B_6_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_B_6_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_B_6_inner), (3, 3)))



print('Player_A_7_first_Strategy')
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
Game_2.layer_input_A_7_inner[np.argmax(Game.layer_input_B_6_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game_2.sigmoid(Game_2.layer_input_A_7_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game_2.sigmoid(Game_2.layer_input_A_7_inner), (3, 3)))

print('Player_A_7_second_Strategy')
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
Game.layer_input_A_7_inner[np.argmax(Game.layer_input_B_6_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_A_7_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_A_7_inner), (3, 3)))


print('Player_B_8_first_Strategy')
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_B_6_inner)] = -10
Game_2.layer_input_B_8_inner[np.argmax(Game.layer_input_A_7_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game_2.sigmoid(Game_2.layer_input_B_8_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game_2.sigmoid(Game_2.layer_input_B_8_inner), (3, 3)))

print('Player_B_8_second_Strategy')
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_B_6_inner)] = -10
Game.layer_input_B_8_inner[np.argmax(Game.layer_input_A_7_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_B_8_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_B_8_inner), (3, 3)))




print('Player_A_9_no_choice_Strategy')
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_A_1_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_B_2_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_A_3_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_B_4_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_A_5_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_B_6_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_A_7_inner)] = -10
Game.layer_input_A_9_inner[np.argmax(Game.layer_input_B_8_inner)] = -10
final = np.zeros(9)
final[ np.argmax(Game.sigmoid(Game.layer_input_A_9_inner))  ] = 1

print(  np.reshape(final, (3, 3))     )
print(  np.reshape(Game.sigmoid(Game.layer_input_A_9_inner), (3, 3)))





