# Código para resolução do ambiente  BipedalWalker-v3 do OpenAI gymnasium
# 300 pontos mínimos para conclusão válida
# A abordagem utilizada é aprendizado por reforço utilizando o algorítmo PPO desenvolvido pela empresa open-source OpenAI
# A implementação disponibiliza opções de processamento singlecore e multicore, utilização da arquitetura CUDA não foi elaborada
# Aconselha-se a utilização de um ambiente python separado, como cuda ou env.
# Caso haja interesse em treinar seu próprio modelo, desaconselha-se a utilização da simulação visual devido ao tempo adicionado


import os
import gymnasium as gym
import pylab
import numpy as np
#import tensorflow as tf
from tensorboardX import SummaryWriter

from tensorflow import compat, random_normal_initializer, where

compat.v1.disable_eager_execution()

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras import backend as K
import copy

from multiprocessing import Process


# HIPERPARAMETROS proprios do PPO na linha 145
# __init__ no final com as opções de processamento single-core, multi-core e carregar modelos salvos

# Para exibir a simulação basta trocar
# self.env = gym.make(env_name)
# para
# self.env = gym.make(env_name, render_mode="human")

LOSS_CLIPPING = 0.2 # hiperparametro, bem estabelecido, desaconselha-se alteração

# definição do ambiente para treinamento e renderização
class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, visualize=False):
        super(Environment, self).__init__()
        #self.env = gym.make(env_name)
        self.env = gym.make(env_name, render_mode="human")
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size

    def run(self):
       
        state, info = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        while True:
            action = self.child_conn.recv()

            if self.is_render and self.env_idx == 0:
                self.env.render()

            state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            state = np.reshape(state, [1, self.state_size])

            if done:
                state, info = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

            self.child_conn.send([state, reward, done, info])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        # cadamas de neuronios
        X = Dense(512, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="tanh")(X)



        self.Actor = Model(inputs = X_input, outputs = output)
        

        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(learning_rate=lr))

    # "perca" da vantagem, diminuição da vantagem, faz com que os saltos da política entre os episódios não sejam muito grandes 
    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    # matematicas da perca, é uma formula geral bem estabelecida
    def gaussian_likelihood(self, actions, pred):
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)
    # predição do próximo estado, utilizado para calcular a recompensa possivel acumulada (lembra do que eu desenhei no quadro aquele dia)
    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        # camadas de neuronio do critico
        V = Dense(512, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(X_input)
        V = Dense(256, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(V)
        V = Dense(64, activation="relu", kernel_initializer=random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        
        #lr é a taxa de aprendizado setada la na classe PPO
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(learning_rate=lr))

    # mesma ideia que a lá de cima, só que para a rede do critico
    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2 # hiperparametro
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
    
# classe principal
class PPOAgent:
    def __init__(self, env_name, model_name=""):
        self.env_name = env_name
        #self.env = gym.make(env_name)  
        self.env = gym.make(env_name, render_mode="human")  # exibe a simulação
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 100 # quantidade de episodios
        self.episode = 0 # episodio inicial
        self.max_average = 0 # media do fitness das instâncias
        self.lr = 0.00009 # taxa de aprendizado, valor atual == indicado pela openAI, mas é bem lento tb
        self.epochs = 10 # epochs por cada batch training
        self.shuffle = True
        self.Training_batch = 256 # 256 instâncias rodando ao mesmo tempo, tipo um multicore
        # cada instância contribui para a atualização da política, mas cada uma subsequente a primeira tem menor importancia


        # Define o diretório de salvamento
        self.save_dir = '/home/andre/Desktop/UNIO/IA/TRAB RNA/REPO DO VIDEO/Modelos'
        
        # Cria o diretório se não existir
        os.makedirs(self.save_dir, exist_ok=True)

        self.optimizer = adam_v2.Adam # otimizador
        self.replay_count = 0 # nao sei
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr)) # cria os logs acessíveis pelo diretório "runs"

        self.scores_, self.episodes_, self.average_ = [], [], [] # inicialização de parametros

        self.Actor = Actor_Model(input_shape=self.state_size, action_space=self.action_size, lr=self.lr, optimizer=self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space=self.action_size, lr=self.lr, optimizer=self.optimizer)

        # nome do arquivo de saida
        self.Actor_name = f"{self.env_name}_PPO_Actor.weights.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.weights.h5"

        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)


    def load(self):
        self.Actor.Actor.load_weights("/home/andre/Desktop/UNIO/IA/TRAB RNA/REPO DO VIDEO/Modelos/actor.h5")
        self.Critic.Critic.load_weights("/home/andre/Desktop/UNIO/IA/TRAB RNA/REPO DO VIDEO/Modelos/critic.h5")

    def act(self, state):
        pred = self.Actor.predict(state)
        low, high = -1.0, 1.0 
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, low, high)
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)
        return action, logp_t
    
# distribuição gaussiana
    def gaussian_likelihood(self, action, pred, log_std):
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):
        gamma = 0.95 # hiperparametro, multiplica pela recompensa (retardar ou acelerar a conversão)
        running_add = 0 
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= (np.std(discounted_r) + 1e-8)
        return discounted_r

    # Estimativa da vantagem, lambda define a importância temporal das ações, valores  próximos a 0 valorizam ações imediatas (TD-Learning) enquanto próximas de 1 valorizam recompensas distantes (monte carlo)
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]
        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    # videozinho
    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        
        y_true = np.hstack([advantages, actions, logp_ts])
        
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    # interface
    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                # Salva o gráfico também no diretório especificado
                plot_path = os.path.join(self.save_dir, self.env_name + ".png")
                pylab.savefig(plot_path)
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save_models()
            SAVING = "SAVING"
            
            #self.lr *= 0.99
            #K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            #K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    
    
    def run_batch(self):
        state, info = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            states, next_states, actions, rewards, dones, logp_ts = [], [], [], [], [], []
            for t in range(self.Training_batch):
                self.env.render()
                # ator escolhe uma ação para o estado atual
                action, logp_t = self.act(state)
                # recebe novo estado, recompensa da ação e se é estado final
                next_state, reward, terminated, truncated, _ = self.env.step(action[0])

                # sinaliza se terminou com sucesso ou falha
                done = terminated or truncated 

                # atualiza e salva o estado atual do ator
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                
                # atualiza os valores atuais das juntas (potencia dos motores)
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
                    
                    state, info = self.env.reset()
                    state = np.reshape(state, [1, self.state_size[0]])

                    # reseta as flags para a próxima iteração, importante se não ele nunca volta do limbo
                    done, score, SAVING = False, 0, ''

            self.replay(states, actions, rewards, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break
        self.env.close()

    # roda igual ao outro (que era pra ser GPU) so que multicore
    def run_multiprocesses(self, num_worker = 4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states = [[] for _ in range(num_worker)]
        next_states = [[] for _ in range(num_worker)]
        actions = [[] for _ in range(num_worker)]
        rewards = [[] for _ in range(num_worker)]
        dones = [[] for _ in range(num_worker)]
        logp_ts = [[] for _ in range(num_worker)]
        score = [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    # prints que aparecem durante o treinamento
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # Salva os modelos finais quando o treinamento terminar
        self.save_models()
        for work in works:
            work.terminate()
            work.join()

    def save_models(self):
            """Salva os modelos finais após o treinamento no diretório especificado"""
            
            self.Actor.Actor.save_weights('/home/andre/Desktop/UNIO/IA/TRAB RNA/REPO DO VIDEO/Modelos/actor.h5', save_format='h5')
            self.Critic.Critic.save_weights('/home/andre/Desktop/UNIO/IA/TRAB RNA/REPO DO VIDEO/Modelos/critic.h5', save_format='h5')


    def test(self, test_episodes):
            # Atualiza os caminhos para carregar do diretório especificado
        self.Actor_name = os.path.join(self.save_dir, f"{self.env_name}_PPO_Actor.h5")
        self.Critic_name = os.path.join(self.save_dir, f"{self.env_name}_PPO_Critic.h5")
        self.load()
        #self.env = gym.make(env_name)
        self.env = gym.make(env_name, render_mode="human")
        for e in range(test_episodes + 1): # 
            
            state, info = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                 
                
                self.env.render() 
                action = self.Actor.Actor.predict(state, verbose=0)[0]
                    
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated # Combine for the done signal
                    
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=True)
                    print("episode: {}/{}, score: {}, average: {:.2f}".format(e, test_episodes, score, average))
                    break
        self.env.close()

if __name__ == "__main__":

    # problema, "ambiente", a ser resolvido
    env_name = 'BipedalWalker-v3'
    agent = PPOAgent(env_name)

    # rodar unicore
    #agent.run_batch()

    #rodar multicore
    #agent.run_multiprocesses(num_worker = 16)
    

    # testa no ambiente
    agent.test(test_episodes=1000)
