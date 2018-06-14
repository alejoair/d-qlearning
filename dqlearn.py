import gym
import numpy as np
import rl
import random
from math import floor
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler

#Configuracion del modelo

max_ep = 30000
epsilon = 0.1
epsilon_min = 0.1
epsilon_decay=0.9999
gamma = 0.99

cantidad = 32
cantidadfit = 500

#--------------------------------

cargar = True                 #Cargar los pesos del entrenamiento
guardar = True                    #Guardar los pesos al entrenar
nombre_archivo = "pesos.h5"
render = False                 #Renderizar el environment
plot = False                         #Ver grafico del estado

memoria = rl.Memoria(cantidadfit * 5) #Crea memoria para el replay
minmax1 = MinMaxScaler()
minmax2 = MinMaxScaler()

#DEFIniR MODELO-------------
#---------------------------------

model = Sequential()
model.add(Dense(16,input_dim=4,activation="hard_sigmoid"))
model.add(Dense(16,activation="hard_sigmoid"))
model.add(Dense(8,activation="hard_sigmoid"))
model.add(Dense(2,activation="linear"))

adam = optimizers.adam(lr=0.0002)
model.compile(optimizer=adam,loss="mse")

#DEFIniR MODELO-------------------
#---------------------------------
try:
    if cargar==True:
        model.load_weights(nombre_archivo) #Cargar los pesos si existe el archivo
        print("Pesos cargados")
        print("\n -----------")
except:

    print("No hay pesos guardados")




env = gym.make("CartPole-v1") #Crea el environment
estado = env.reset()
estado = np.reshape(estado,[1,4])
accion = 0
prediccion = 0
r = 0
ep = 0
x = []
y1,y2,y3,y4 = [],[],[],[]
rarray = r

while True:

    # SELECIONAR LA ACCION------
    # --------------------------
    if random.uniform(0,1)<epsilon:

        accion = random.randint(0,1)

    else:
        acc = model.predict(estado)
        accion = np.argmax(acc[0])

    # SELECIONAR LA ACCION------
    # --------------------------

    #Interactua con el AMbiente
    estado_,reward,done,info = env.step(accion)
    r += reward
    y1.append(estado_[0]) #
    y2.append(estado_[1]) #Para pruebas
    y3.append(estado_[2]) #
    y4.append(estado_[3]) #

    reward = reward - (abs(estado_[2]) + abs(estado_[3]) + abs(estado_[0])) # Da valor al reward

    estado_ = np.reshape(estado_,[1,4])


    if done == True:
        pass



    #Genera el target para el entrenamiento de la red
    p = model.predict(estado)
    p_max = model.predict(estado_)

    p_max = np.amax(p_max)

    target = reward + gamma*p_max
    p[0][accion] = target

    #--------------------------------------------------

    memoria.agregar(estado, accion, reward, estado_,p)  # Guarda en la memoria los valores
    estado = estado_ #Actualiza el estado


    mem = memoria.leer(cantidad)
    a = mem[:cantidad,0]
    a_ = np.zeros([1,4])
    for i,v in enumerate(a):
        a_ = np.vstack([a[i][0],a_])
    a_ = a_[0:-1]

    tar = mem[:cantidad,4]
    tar_ = np.zeros([1, 2])
    for i,v in enumerate(tar):
        tar_ = np.vstack([tar[i][0],tar_])
    tar_ = tar_[0:-1]

    if render==True: #------------Renderiza el environment
        env.render()

#.------------------------------Si termina el episodio---------------

    if (done == True):
        ep += 1
        if plot:

            plt.subplot(411)
            plt.plot(y1, label="Posicion Carro")
            plt.legend()
            plt.subplot(412) #para pruebas
            plt.plot(y2, label="Velocidad Carro")
            plt.legend()
            plt.subplot(413)
            plt.plot(y3,label="Angulo Palo")
            plt.legend()
            plt.subplot(414)
            plt.plot(y4,label="Velocidad Palo")
            plt.legend()
            plt.show()

                                    #Plot del estado
        y1,y2,y3,y4 = [],[],[],[]

        if ep>max_ep:
            print("Entrenamiento terminado en ", ep, " Episodios")         #Termina el entrenamiento si supera el maximo de episodios
            break
        if floor(ep % 100) == 0 and guardar==True:

            print("Pesos Guardados")
            model.save_weights(nombre_archivo)


        if epsilon> epsilon_min:
            epsilon *= epsilon_decay

        model.fit(a_, tar_, verbose=0)

        print("Episodio----- ", ep, " Reward------- ",r," Epsilon------- ", epsilon)
        r = 0
        estado = env.reset()
        estado = np.reshape(estado,[1,4])