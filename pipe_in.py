from sys import byteorder
from env.SrsRanEnv import SrsRanEnv
import multiprocessing as mp
from Config import Config
from A3CAgent import A3CAgent
from multiprocessing import shared_memory
import numpy as np


ACTOR_IN = '/tmp/actor_in'
ACTOR_OUT = '/tmp/actor_out'
REWARD_IN = '/tmp/return_in'

class Coordinator():
    def __init__(self):
        self.total_agents = 8
        self.verbose = 0

        # validity byte
        # observation is: noise, beta, bsr all are integers32
        try:
            shm_observation = shared_memory.SharedMemory(create = True,  name = 'observation', size = (16) * self.total_agents)
        except Exception:
            shm_observation = shared_memory.SharedMemory(create = False, name = 'observation', size = (16) * self.total_agents)

        nd_array = np.ndarray(shape=(4 * self.total_agents), dtype=np.int32, buffer=shm_observation.buf)
        nd_array[:] = np.full(shape=(4 * self.total_agents), fill_value=0)

        try:
            shm_action = shared_memory.SharedMemory(create = True,  name = 'action', size = (12) * self.total_agents)
        except Exception:
            shm_action = shared_memory.SharedMemory(create = False, name = 'action', size = (12) * self.total_agents)

        nd_array = np.ndarray(shape=(3 * self.total_agents), dtype=np.int32, buffer=shm_action.buf)
        nd_array[:] = np.full(shape=(3 * self.total_agents), fill_value=0)

        try:
            shm_reward = shared_memory.SharedMemory(create = True,  name = 'result', size = (16) * self.total_agents)
        except Exception:
            shm_reward = shared_memory.SharedMemory(create = False, name = 'result', size = (16) * self.total_agents)    

        nd_array = np.ndarray(shape=(4 * self.total_agents), dtype=np.int32, buffer=shm_reward.buf)
        nd_array[:] = np.full(shape=(4 * self.total_agents), fill_value=0)                

        self.config = self.get_environment_config()
        self.processes_started_successfully = mp.Value('i', 0)
        
        self.a3c_agent = A3CAgent(self.config, self.total_agents)

        self.sched_proc = mp.Process(target=self.rcv_obs_send_act_func, name= 'scheduler_intf')
        self.decod_proc = mp.Process(target=self.rcv_return_func, name='decoder_intf')

    def get_environment_config(self) -> Config:
        config = Config()
        config.seed = 1
        config.environment = SrsRanEnv(title = 'SRS RAN Environment', verbose=self.verbose)
        config.num_episodes_to_run = 1e3
        config.save_results = True
        config.results_file_path = '/home/naposto/phd/nokia/data/csv_42/results_wo_pretrained.csv'

        config.save_weights = True
        config.save_weights_period = 100
        config.weights_file_path = '/home/naposto/phd/nokia/data/csv_42/weights_wo_pretrained.h5'
        
        config.load_initial_weights = False
        config.initial_weights_path = '/home/naposto/phd/nokia/data/csv_41/beta_all_noise_all_entropy_0.1_model.h5'

        config.hyperparameters = {
            'Actor_Critic_Common': {
                'learning_rate': 1e-3,
                'linear_hidden_units': [5, 32, 64, 100],
                # 'linear_hidden_units': [5, 32],
                'num_actor_outputs': 2,
                'final_layer_activation': ['softmax', 'softmax', None],
                'normalise_rewards': False,
                'add_extra_noise': False,
                'batch_size': 64,
                'local_update_period': 1, # in episodes
                'include_entropy_term': True,
                'entropy_beta': 0.1,
                'entropy_contrib_prob': 0.995,
                'Actor': {
                    'linear_hidden_units': [100, 40]
                    # 'linear_hidden_units': [25]
                },
                'Critic': {
                    'linear_hidden_units': [16, 4]
                    # 'linear_hidden_units': [3]
                }
            }
        }

        return config    

    def start(self):
        self.sched_proc.start()
        self.decod_proc.start()
        self.a3c_agent.run_n_episodes(self.processes_started_successfully)
        self.sched_proc.join()
        self.decod_proc.join()

    def get_action(self):
        return 

    def rcv_return_func(self):
        shm_reward = shared_memory.SharedMemory(create = False,  name = 'result')
        self.reward_nd_array = np.ndarray(
            shape=(4 * self.total_agents),
            dtype= np.int32,
            buffer = shm_reward.buf
        )
        while (self.processes_started_successfully.value == 0):
            pass
        print('Receive result thread waiting for all processes to start...OK')
        is_file_open = False
        while (not is_file_open):
            try:
                with open(REWARD_IN, mode='rb') as file_read:
                    is_file_open = True
                    while (True):
                        content = file_read.read(16)
                        if (len(content) <= 0):
                            print('EOF')
                            break
                        tti = int.from_bytes(content[0:2], "little")
                        rnti = int.from_bytes(content[2:4], "little")
                        dec_time = int.from_bytes(content[4:8], "little")
                        crc = int.from_bytes(content[8:9], "little")
                        dec_bits = int.from_bytes(content[12:], "little")

                        result_buffer = [crc, dec_time, dec_bits]
                        agent_idx = tti % self.total_agents
                        if (self.verbose == 1):
                            print('Res {} - {}'.format(agent_idx, result_buffer))
                        self.reward_nd_array[agent_idx * 4: (agent_idx + 1) * 4] = np.array([1, *result_buffer], dtype=np.int32)
            except FileNotFoundError as e:
                pass
                


    def rcv_obs_send_act_func(self):
        shm_observation = shared_memory.SharedMemory(create = False,  name = 'observation')
        shm_action = shared_memory.SharedMemory(create = False,  name = 'action')

        self.observation_nd_array = np.ndarray(
            shape=(4 * self.total_agents), 
            dtype= np.int32, 
            buffer = shm_observation.buf)

        
        self.action_nd_array = np.ndarray(
            shape=(3 * self.total_agents),
            dtype= np.int32,
            buffer = shm_action.buf
        )

        while (self.processes_started_successfully.value == 0):
            pass
        print('Receive obs thread waiting for all processes to start... OK')
        is_actor_in_open = False
        while (not is_actor_in_open):
            try:                
                with open(ACTOR_IN, mode='rb') as file_read:
                    is_actor_in_open = True
                    with open(ACTOR_OUT,  mode='wb') as file_write:
                        while (True):
                            content = file_read.read(16)
                            if (len(content) <= 0):
                                print('EOF')
                                break
                            tti  = int.from_bytes(content[0:2], "little")
                            rnti = int.from_bytes(content[2:4], "little")
                            bsr =  int.from_bytes(content[4:8], "little")
                            noise =  int.from_bytes(content[8:12], "little", signed = True)
                            beta = int.from_bytes(content[12:], "little")
                            
                            agent_idx = tti % self.total_agents
                            observation = [noise, beta, bsr]
                            if (self.verbose == 1):
                                print('Obs {} - {}'.format(agent_idx, observation))
                            self.observation_nd_array[agent_idx * 4: (agent_idx + 1) * 4] = np.array([1, *observation], dtype=np.int32)

                            while self.action_nd_array[agent_idx * 3]  == 0:
                                pass

                            self.action_nd_array[agent_idx * 3 ] = 0
                            mcs, prb = self.action_nd_array[agent_idx * 3 + 1].item(), self.action_nd_array[agent_idx * 3 + 2].item()
                            if (self.verbose == 1):
                                print('Act {} - {}'.format(agent_idx, [mcs, prb]))

                            action_mcs = mcs.to_bytes(1, byteorder="little")
                            action_prb = prb.to_bytes(1, byteorder="little")
                            ext_byte_arr = action_mcs +  action_prb
                            file_write.write(ext_byte_arr)
                            file_write.flush()
            except FileNotFoundError as e:
                if (is_actor_in_open):
                    raise e
                pass

if __name__== '__main__':
    coordinator = Coordinator()
    coordinator.start()
