from sys import byteorder
from SrsRanEnv import SrsRanEnv
import multiprocessing as mp

ACTOR_IN = '/tmp/actor_in'
ACTOR_OUT = '/tmp/actor_out'
REWARD_IN = '/tmp/return_in'

class Coordinator():
    def __init__(self):
        self.total_agents = 8

        self.lock = mp.Lock()

        self.obs_q = [mp.Queue(maxsize=1) for _ in range(self.total_agents)]
        self.rew_q = [mp.Queue(maxsize=1) for _ in range(self.total_agents)]
        self.act_q = mp.Queue(maxsize=1)
        
        self.agents = [SrsRanEnv(self.obs_q[i], self.act_q, self.rew_q[i], title = 'worker_' + str(i)) for i in range(self.total_agents)]

        self.sched_proc = mp.Process(target=self.rcv_obs_send_act_func, name= 'scheduler_intf')
        self.decod_proc = mp.Process(target=self.rcv_return_func, name='decoder_intf')
        

    def start(self):
        self.sched_proc.start()
        self.decod_proc.start()
        [worker_proc.start() for worker_proc in self.agents]
        self.sched_proc.join()
        self.decod_proc.join()

    def get_action(self):
        return 

    def rcv_return_func(self):
        is_file_open = False
        while (not is_file_open):
            try:
                with open(REWARD_IN, mode='rb') as file_read:
                    if_file_open = True
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

                        reward_buffer = [tti, crc, dec_time, dec_bits]
                        agent_idx = tti % self.total_agents
                        self.rew_q[agent_idx].put(reward_buffer)
            except FileNotFoundError as e:
                pass
                


    def rcv_obs_send_act_func(self):
        with open(ACTOR_IN, mode='rb') as file_read, \
            open(ACTOR_OUT,  mode='wb') as file_write:
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
                
                observation = [tti, noise, beta, bsr]

                agent_idx = tti % self.total_agents
                self.obs_q[agent_idx].put(observation)

                mcs, prb = self.act_q.get(block=True)
                # mcs, prb = 10, 10

                action_mcs = mcs.to_bytes(1, byteorder="little")
                action_prb = prb.to_bytes(1, byteorder="little")
                ext_byte_arr = action_mcs +  action_prb
                file_write.write(ext_byte_arr)
                file_write.flush()

if __name__== '__main__':
    coordinator = Coordinator()
    coordinator.start()
