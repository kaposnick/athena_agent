import multiprocessing as mp
import argparse

from common_utils import MODE_SCHEDULING_ATHENA, MODE_SCHEDULING_SRS, MODE_SCHEDULING_RANDOM
from srsran_env import SrsRanEnv
from config import Config
from agent_factory import AgentFactory
from coordinator import Coordinator
from log_process import LogProcess


coordinator = None
agent_factory = None
stop_flag   = mp.Value('i', 0)

def exit_gracefully(signal, frame):
    stop_flag.value = 1
    if (agent_factory is not None):
        agent_factory.kill()
    
    if (coordinator is not None):
        coordinator.kill()

def get_config() -> Config:
    parser = argparse.ArgumentParser(description='srsENB AI scheduler implementation')
    parser.add_argument('-m', '--mode', choices=['athena', 'srs', 'random'], required=True, dest='mode')
    parser.add_argument('-r', '--results', required=True, dest='path_results')
    parser.add_argument('--actions', type=int, choices=range(1,3), dest='actions')
    parser.add_argument('--actor_weights', dest='actor_weights')
    parser.add_argument('--critic_weights', dest='critic_weights')
    parser.add_argument('--verbose', type=int, choices=range(0,2), dest='verbose', default=0)
    
    scheduling_mode = None
    path_results = None
    path_actor_weights = None
    path_critic_weights = None
    action_size = None
    
    args = parser.parse_args()
    print(args)
    path_results = args.path_results
    mode = args.mode
    verbose = 0
    if (mode == 'athena'):
        scheduling_mode = MODE_SCHEDULING_ATHENA
        action_size = args.actions
        if (action_size == 1):
            path_actor_weights = args.actor_weights
            path_critic_weights = args.critic_weights
        elif (action_size == 2):
            path_actor_weights = args.actor_weights
            path_critic_weights = args.critic_weights
    elif (mode == 'srs'):
        scheduling_mode = MODE_SCHEDULING_SRS
        action_size = 2
    elif (mode == 'random'):
        scheduling_mode = MODE_SCHEDULING_RANDOM
        action_size = 2

    config = Config()
    config.context_size = 2
    config.action_size = action_size
    config.scheduling_mode = scheduling_mode
    config.load_weights = scheduling_mode == MODE_SCHEDULING_ATHENA
    config.actor_path = path_actor_weights
    config.critic_path = path_critic_weights
    config.result_path = path_results
    config.verbose     = verbose
    config.environment = SrsRanEnv(
            context_size=2, action_size=action_size, 
            penalty=1, title = 'srsRAN', verbose=verbose,
            decode_deadline=3000, scheduling_mode=scheduling_mode)
    return config

if __name__== '__main__':
    import signal
    signal.signal(signal.SIGINT , exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    config:Config = get_config()

    results_queue = mp.Queue()
    log_process = LogProcess(
        log_queue=results_queue, 
        scheduling_mode=config.scheduling_mode, 
        log_file=config.result_path,
        stop_flag=stop_flag)
    log_process.start()

    total_agents = 8    
    cond_observations = [mp.Condition() for _ in range(total_agents)]
    cond_actions      = [mp.Condition() for _ in range(total_agents)]
    cond_verify_action= [mp.Condition() for _ in range(total_agents)]
    cond_rewards      = [mp.Condition() for _ in range(total_agents)]
    agent_coordination_lock = mp.Value('i', 0)

    coordinator = Coordinator(
        observation_locks=cond_observations, 
        action_locks=cond_actions, 
        reward_locks=cond_rewards, 
        verify_action_locks=cond_verify_action,
        agent_coordination_lock=agent_coordination_lock,
        verbose=config.verbose
    )
    coordinator.start()

    inputs = []
    for idx in range(total_agents):
        input = {}
        input['cond_observation'] = cond_observations[idx]
        input['cond_action']      = cond_actions[idx]
        input['cond_verify_action']     = cond_verify_action[idx]
        input['cond_reward'] = cond_rewards[idx]
        inputs.append(input)
    agent_factory = AgentFactory(
        config=config, 
        agent_coordination_lock=agent_coordination_lock,
        stop_flag=stop_flag)
    agent_factory.start(
        inputs=inputs,
        results_queue=results_queue
    )

    log_process.join()


    