from srsran_env import SrsRanEnv
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

FROM_MAC_CONTEXT = '/tmp/actor_in'
TO_MAC_ACTION    = '/tmp/actor_out'
FROM_MAC_VERIFY  = '/tmp/verify_action'
FROM_PHY_REWARD  = '/tmp/return_in'

class Coordinator():
    def __init__(self,
                 observation_locks,  action_locks,  reward_locks, verify_action_locks,
                 observation_size=6, action_size=3, reward_size=9, verify_action_size=2,
                 agent_coordination_lock=None, verbose=0):
        self.total_agents = 8
        self.cond_observations = observation_locks
        self.cond_actions      = action_locks
        self.cond_rewards      = reward_locks
        self.cond_verify_action = verify_action_locks
        self.observation_size=observation_size
        self.action_size=action_size
        self.reward_size=reward_size
        self.verify_action_size=verify_action_size
        self.verbose = 0

        self.sched_proc = mp.Process(target=self.func_scheduler, name= 'scheduler_intf')        
        self.decod_proc = mp.Process(target=self.func_decoder, name='decoder_intf')
        
        self.agent_coordination_lock = agent_coordination_lock

        self.get_observation_memory()
        self.get_action_memory()
        self.get_verify_action_memory()
        self.get_reward_memory()

    def kill(self):
        print('Killing Coordinator')        
        if (self.decod_proc.is_alive()):
            self.decod_proc.kill()
            self.decod_proc.join()
        if (self.sched_proc.is_alive()):
            self.sched_proc.kill()
            self.sched_proc.join()    

    def start(self):
        self.sched_proc.start()
        self.decod_proc.start()

    def wait_agents_to_finish_init(self):
        if (self.agent_coordination_lock is not None):
            while (self.agent_coordination_lock.value == 0):
                pass
    
    def func_decoder(self, reward_packet_size=32):
        shm_reward = shared_memory.SharedMemory(create = False,  name = 'result')
        reward_nd_array = np.ndarray(shape=(self.reward_size * self.total_agents), dtype= np.int32, buffer = shm_reward.buf)
        self.wait_agents_to_finish_init()    
        print('Receive result thread waiting for all processes to start...OK')
        is_file_open = False
        while (not is_file_open):
            try:
                with open(FROM_PHY_REWARD, mode='rb') as file_read:
                    is_file_open = True
                    print('Opening receive reward socket...')
                    while (True):
                        content = file_read.read(reward_packet_size)
                        if (len(content) <= 0):
                            print('EOF')
                            break
                        tti      = int.from_bytes(  content[0:2], "little")
                        rnti     = int.from_bytes(  content[2:4], "little")
                        dec_time = int.from_bytes(  content[4:8],   "little")
                        crc      = int.from_bytes(  content[8:9],   "little")
                        dec_bits = int.from_bytes(  content[12:16], "little")
                        mcs      = int.from_bytes(  content[16:18], "little")
                        prb      = int.from_bytes(  content[18:20], "little")
                        snr      = int.from_bytes(  content[20:24], "little")
                        noise    = int.from_bytes(  content[24:28], "little")
                        snr_custom = int.from_bytes(content[28:32], "little")

                        result_buffer = np.array([tti, crc, dec_time, dec_bits, mcs, prb, snr, noise, snr_custom], dtype = np.int32)
                        agent_idx = tti % self.total_agents
                        if (self.verbose == 1):
                            print('Res {} - {}'.format(agent_idx, result_buffer))
                        cond_reward = self.cond_rewards[agent_idx]
                        result_buffer[0] = 1
                        with cond_reward:
                            reward_nd_array[agent_idx * self.reward_size: (agent_idx + 1) * self.reward_size] = result_buffer
                            cond_reward.notify()
            except FileNotFoundError as e:
                pass               


    def func_scheduler(self, context_packet_size=16, verify_packet_size=4):
        shm_observation = shared_memory.SharedMemory(create = False,  name = 'observation')
        shm_action = shared_memory.SharedMemory(create = False,  name = 'action')        
        shm_verify_action = shared_memory.SharedMemory(create = False, name = 'verify_action')
        observation_nd_array = np.ndarray(shape=(self.observation_size * self.total_agents), dtype= np.int32, buffer = shm_observation.buf)        
        action_nd_array = np.ndarray(shape=(self.action_size * self.total_agents), dtype= np.int32, buffer = shm_action.buf)
        verify_action_nd_array = np.ndarray(shape=(self.verify_action_size * self.total_agents), dtype= np.int32, buffer = shm_verify_action.buf)
        
        self.wait_agents_to_finish_init()
        print('Receive obs thread waiting for all processes to start... OK')
        is_actor_in_open = False
        while (not is_actor_in_open):
            try:                
                with open(FROM_MAC_CONTEXT, mode='rb') as file_read:
                    is_actor_in_open = True
                    is_verify_action_open = False
                    while (not is_verify_action_open):
                        try:
                            with open(FROM_MAC_VERIFY, mode = 'rb') as verify_action_fd:
                                is_verify_action_open = True
                                with open(TO_MAC_ACTION,  mode='wb') as file_write:
                                    print('Opening receive context socket...')
                                    while (True):
                                        content = file_read.read(context_packet_size)
                                        if (len(content) <= 0):
                                            print('EOF')
                                            break
                                        tti  = int.from_bytes(content[0:2], "little")
                                        rnti = int.from_bytes(content[2:4], "little")
                                        bsr =  int.from_bytes(content[4:8], "little")
                                        snr =  int.from_bytes(content[8:12], "little", signed = True)
                                        beta = int.from_bytes(content[12:14], "little")
                                        gain = int.from_bytes(content[14:], "little")
                                        
                                        agent_idx = tti % self.total_agents
                                        observation = np.array([1, tti, beta, snr, bsr, gain], dtype = np.int32)
                                        if (self.verbose == 1):
                                            print('Obs {} - {}'.format(agent_idx, observation))
                                        cond_observation   = self.cond_observations[agent_idx]
                                        cond_action        = self.cond_actions[agent_idx]

                                        with cond_observation:
                                            observation_nd_array[agent_idx * self.observation_size: (agent_idx + 1) * self.observation_size] = observation
                                            cond_observation.notify()

                                        with cond_action:
                                            while action_nd_array[agent_idx * self.action_size]  == 0:
                                                cond_action.wait(0.001)

                                        action_nd_array[agent_idx * self.action_size ] = 0
                                        mcs, prb = action_nd_array[agent_idx * self.action_size + 1].item(), action_nd_array[agent_idx * self.action_size + 2].item()
                                        if (self.verbose == 1):
                                            print('Act {} - {}'.format(agent_idx, [tti, mcs, prb]))

                                        action_mcs = mcs.to_bytes(1, byteorder="little")
                                        action_prb = prb.to_bytes(1, byteorder="little")
                                        ext_byte_arr = action_mcs +  action_prb
                                        file_write.write(ext_byte_arr)
                                        file_write.flush()

                                        cond_verify_action = self.cond_verify_action[agent_idx]
                                        verify_action_content = verify_action_fd.read(verify_packet_size)
                                        if (len(verify_action_content) < 0):
                                            print('EOF')
                                            break
                                        action_verified = int.from_bytes(verify_action_content[0: 4], "little")
                                        with cond_verify_action:
                                            verify_action_nd_array[agent_idx * self.verify_action_size: (agent_idx + 1) * self.verify_action_size] = np.array([1, action_verified], dtype = np.int32)
                                            cond_verify_action.notify()
                        except FileNotFoundError as e:
                            if (is_actor_in_open and is_verify_action_open):
                                raise e
                            pass
            except FileNotFoundError as e:
                if (is_actor_in_open):
                    raise e
                pass

    def get_memory_buffer(self, buffer_size_per_agent, buffer_name):
        int_size = 4
        size = buffer_size_per_agent * int_size * self.total_agents
        try:
            shm = shared_memory.SharedMemory(create = True,  name=buffer_name, size=size)
        except Exception:
            shm = shared_memory.SharedMemory(create = False, name=buffer_name, size=size)
        nd_array = np.ndarray(shape=(buffer_size_per_agent * self.total_agents), dtype=np.int32, buffer=shm.buf)
        nd_array[:] = np.full(shape=(buffer_size_per_agent * self.total_agents), fill_value=0)
        return nd_array

    def get_observation_memory(self):
        return self.get_memory_buffer(self.observation_size, 'observation')

    def get_action_memory(self):
        return self.get_memory_buffer(self.action_size, 'action')

    def get_verify_action_memory(self):
        return self.get_memory_buffer(self.verify_action_size, 'verify_action')

    def get_reward_memory(self):
        return self.get_memory_buffer(self.reward_size, 'result')