import random
import gym
import socket 
import time 
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pickle,os
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

MODEL_PATH = './saved/model.h5'
TABLE_PATH = './saved/table.pkl'

class DQNSolver:

    def load_model_and_table(self):
        if not os.path.exists(MODEL_PATH):
            return False
        self.model = keras.models.load_model(MODEL_PATH)
        with open(TABLE_PATH,'rb') as f:
            self.memory = pickle.load(f)
        print('loaded')
        # print(self.memory)
        return True
    
    def save_model_and_table(self):
        self.model.save(MODEL_PATH)
        with open(TABLE_PATH,'wb') as f:
            pickle.dump(self.memory,f)

    def __init__(self, observation_space, action_space,israndom = True): 
        self.exploration_rate = EXPLORATION_MIN
        self.israndom = israndom
        self.action_space = action_space
        if not self.load_model_and_table():
            self.exploration_rate = EXPLORATION_MAX

            self.action_space = action_space
            self.memory = deque(maxlen=MEMORY_SIZE)
            print("OBSERVATION SPACE = " , observation_space)

            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
            print('created')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate and self.israndom:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)                      
    port = 12345

    # connection to hostname on the port.
    s.connect(('192.168.43.110', port))             

    print ("Connected to environment")                   

    score_logger = ScoreLogger(ENV_NAME)
    observation_space = 4
    action_space = 4
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    
    for _ in range(10):

        run += 1
        func = dict() 
        func["function"] = "render" 

        s.send(pickle.dumps(func)) 
        state = pickle.loads(s.recv(1024))["state"] 

        step = 0
        while True:
            startTime = time.time() 
            step += 1
            state = np.reshape(state, [1, observation_space]) 
            action = dqn_solver.act(state)
            # action = 0    
            func["function"] = "step" 
            func["action"] = action 
            s.send(pickle.dumps(func)) 
            recieved = pickle.loads(s.recv(1024)) 
            state_next, reward, terminal = recieved["state"],recieved["reward"],recieved["terminal"]  
            print (state, reward, terminal) 
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            print (state_next.shape) 
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal or time.time() - startTime > 30: 
                print("Run: "+str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                score_logger.add_score(step, run)
                break   
            dqn_solver.experience_replay()
    dqn_solver.save_model_and_table()

if __name__ == "__main__":

    cartpole()