import multiprocessing as mp
from agent_harq import HarqAgent
from agent_main import MainAgent
from config import Config
from srsran_env import SrsRanEnv


class AgentFactory():
    def __init__(self, config: Config, agent_coordination_lock, stop_flag: mp.Value) -> None:
        self.total_agents = 8
        self.main_agent  = None
        self.harq_agents = [None] * self.total_agents
        self.actor_memory_name = 'model_actor'
        self.critic_memory_name = 'model_critic'
        self.agent_coordination_lock = agent_coordination_lock
        self.context_size    = config.context_size
        self.action_size   = config.action_size
        self.scheduling_mode = config.scheduling_mode
        self.environment:SrsRanEnv   = config.environment
        self.load_weights  = config.load_weights
        self.actor_path    = config.actor_path
        self.critic_path   = config.critic_path
        self.stop_flag     = stop_flag

    def start(self, inputs = None, results_queue=None):
        main_initialized = mp.Value('i', 0)
        self.main_agent = MainAgent(
            context_size=self.context_size, action_size=self.action_size,
            load_initial_weights=self.load_weights,
            main_agent_initialized=main_initialized,
            stop_flag=self.stop_flag,
            actor_initial_weights_path=self.actor_path,
            critic_initial_weights_path=self.critic_path
        )
        self.main_agent.start()
        while (main_initialized.value == 0):
            pass
        
        print('Main Agent started successfully')
        harq_agents_initialized = mp.Value('i', 0)
        for worker_num in range(self.total_agents):
            import copy
            worker_environment = copy.deepcopy(self.environment)
            worker_environment.presetup(inputs[worker_num])
            worker = HarqAgent(
                environment=worker_environment,
                worker_num=worker_num,
                total_workers=self.total_agents,                
                context_size=self.context_size, action_size=self.action_size,
                successfully_started_worker=harq_agents_initialized,
                results_queue=results_queue, scheduling_mode=self.scheduling_mode,
                actor_memory_name=self.actor_memory_name, critic_memory_name=self.critic_memory_name
            )
            self.harq_agents[worker_num] = worker
            worker.start()
        while (harq_agents_initialized.value < self.total_agents):
            pass
        print('HARQ Agents started successfully')
        self.agent_coordination_lock.value = 1

    def kill(self):
        self.stop_flag.value = 1
        print('Killing Main Agent')
        if (self.main_agent is not None and self.main_agent.is_alive()):
            self.main_agent.kill()
            self.main_agent.join()

        print('Killing HARQ Agents')
        for worker_process in self.harq_agents:
            if (worker_process is not None and worker_process.is_alive()):
                worker_process.kill()
                worker_process.join()
        

        