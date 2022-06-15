from sys import byteorder
from env.SrsRanEnv import SrsRanEnv
import multiprocessing as mp
from Config import Config
from A3CAgent import A3CAgent


ACTOR_IN = '/tmp/actor_in'
ACTOR_OUT = '/tmp/actor_out'
REWARD_IN = '/tmp/return_in'

class Coordinator():
    def __init__(self):
        self.total_agents = 8

        self.obs_q = [mp.Queue(maxsize=1) for _ in range(self.total_agents)]
        self.rew_q = [mp.Queue(maxsize=1) for _ in range(self.total_agents)]
        self.act_q = mp.Queue(maxsize=1)

        self.config = self.get_environment_config()
        
        self.a3c_agent = A3CAgent(self.config, self.total_agents)

        self.sched_proc = mp.Process(target=self.rcv_obs_send_act_func, name= 'scheduler_intf')
        self.decod_proc = mp.Process(target=self.rcv_return_func, name='decoder_intf')

    def get_environment_config(self) -> Config:
        config = Config()
        config.seed = 1
        config.environment = SrsRanEnv(title = 'SRS RAN Environment')
        config.num_episodes_to_run = 100e3
        config.save_results = True
        config.results_file_path = '/home/naposto/phd/nokia/data/csv_42/results.csv'

        config.save_weights = True
        config.save_weights_period = 100
        config.weights_file_path = '/home/naposto/phd/nokia/data/csv_42/weights.h5'
        
        config.load_initial_weights = True
        config.initial_weights_path = '/home/naposto/phd/nokia/data/csv_41/beta_all_noise_all_entropy_0.1_model.h5'

        config.hyperparameters = {
            'Actor_Critic_Common': {
                'learning_rate': 1e-4,
                'linear_hidden_units': [5, 32, 64, 100],
                'num_actor_outputs': 2,
                'final_layer_activation': ['softmax', 'softmax', None],
                'normalise_rewards': False,
                'add_extra_noise': False,
                'batch_size': 64,
                'include_entropy_term': True,
                'local_update_period': 1, # in episodes
                'entropy_beta': 0.1,
                'Actor': {
                    'linear_hidden_units': [100, 40]
                },
                'Critic': {
                    'linear_hidden_units': [16, 4]
                }
            }
        }

        return config    

    def start(self):
        self.sched_proc.start()
        self.decod_proc.start()
        self.a3c_agent.run_n_episodes(self.obs_q, self.act_q, self.rew_q)
        self.sched_proc.join()
        self.decod_proc.join()

    def get_action(self):
        return 

    def rcv_return_func(self):
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

                        reward_buffer = [crc, dec_time, dec_bits]
                        print('Rew {}'.format([tti, *reward_buffer]))
                        agent_idx = tti % self.total_agents
                        self.rew_q[agent_idx].put(reward_buffer)
            except FileNotFoundError as e:
                pass
                


    def rcv_obs_send_act_func(self):
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
                            
                            observation = [noise, beta, bsr]
                            print('Obs {}'.format([tti, *observation]))

                            agent_idx = tti % self.total_agents
                            self.obs_q[agent_idx].put(observation)

                            mcs, prb = self.act_q.get(block=True)

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
