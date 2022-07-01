#!/usr/bin/python3

from ast import If
from random import random
import subprocess, os
import signal
from time import sleep
from sys import stderr
import numpy as np

iperf_server_address = '172.16.0.2'
iperf_duration = '0.1' # in seconds
spgw_if_address = '172.18.0.1'
num_of_ues = 1
csv_report_period = 0.05

proc_iperf_server = None
proc_enb = None
proc_epc = None
proc_iperf_client = None

loc_exec_ip = '/sbin/ip'
loc_exec_iperf = '/usr/bin/iperf'
loc_exec_echo = '/bin/echo'

def signal_handler(signal, frame):
    processes = [proc_iperf_client, proc_iperf_server, proc_epc, proc_enb]
    for process in processes:
        if process is None:
            continue
        process.kill()
        process.join()
    exit(2)
    
signal.signal(signal.SIGINT, signal_handler)

import os

def start_iperf_server_process():
    return subprocess.Popen([loc_exec_iperf, '-s', '-u'], stdout=subprocess.DEVNULL)

def start_epc_process():
    cmd = ['/home/naposto/tools/srsRAN/bin/srsepc']
    cmd.append('/home/naposto/.config/srsran/epc.conf')
    cmd.append('--spgw.sgi_if_addr={}'.format(spgw_if_address))
    cmd.append('--hss.db_file')
    cmd.append('/home/naposto/.config/srsran/user_db.csv')
    proc_epc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    while True:
        output = proc_epc.stdout.readline()
        # print(output)
        if ('SP-GW Initialized' in output):
            print('EPC Initialized successfully...')
            break

    return proc_epc

def start_agent_process(initial_seed, 
                        num_of_episodes,
                        agent_results_file,
                        pretrained = False,
                        initial_weights_file = None):
    cmd = []
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/naposto/.local/lib/python3.8/site-packages' # it's the gym package
    cmd.append('/usr/bin/python3')
    cmd.append('/home/naposto/phd/nokia/agents/pipe_in.py')
    cmd += [str(initial_seed), str(num_of_episodes), agent_results_file, str(int(pretrained))]
    if (pretrained):
        cmd.append(initial_weights_file)
    output = '/home/naposto/phd/nokia/data/csv_46/agent_output.txt'
    with open(output, 'w') as f:
        proc_agent = subprocess.Popen(cmd, stdout=f, stderr=f, env = env, universal_newlines=True)
    return proc_agent

def start_enb_process( 
        beta_factor, enb_log_file, csv_report_period, enb_csv_file, 
        scheduler, 
        agent_seed = None, agent_episodes = None, agent_results_file = None, agent_pretrained = None, agent_pretrained_weights = None,
        tx_port = 2000, rx_port = 2001):
    cmd = ['/home/naposto/tools/srsRAN/bin/srsenb']
    cmd.append('/home/naposto/.config/srsran/enb.conf')
    cmd.append('--enb_files.sib_config')
    cmd.append('/home/naposto/.config/srsran/sib.conf')
    cmd.append('--enb_files.rr_config')
    cmd.append('/home/naposto/.config/srsran/rr.conf')
    cmd.append('--enb_files.rb_config')
    cmd.append('/home/naposto/.config/srsran/rb.conf')
    cmd.append('--log.phy_level=warning')
    cmd.append('--log.filename={}'.format(enb_log_file))
    cmd.append('--expert.metrics_csv_enable=false')
    cmd.append('--expert.metrics_period_secs={}'.format(csv_report_period))
    cmd.append('--expert.metrics_csv_filename={}'.format(enb_csv_file))
    cmd.append('--expert.pusch_beta_factor={}'.format(beta_factor))
    cmd.append('--scheduler.policy={}'.format(scheduler))
    cmd.append('--rf.device_name=zmq')
    cmd.append('--rf.device_args=fail_on_disconnect=true,tx_port=tcp://*:{},rx_port=tcp://localhost:{},id=enb,base_srate=23.04e6'.format(tx_port, rx_port))

    proc_enb = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    proc_agent = None
    if (scheduler == "time_sched_ai"):
        proc_agent = start_agent_process(
            agent_seed, agent_episodes, agent_results_file, 
            agent_pretrained, agent_pretrained_weights)
        

    while True:
        output = proc_enb.stdout.readline()
        # print(output)
        if ('==== eNodeB started' in output.strip()):
            cmd = ['ps', '-cT', '-p', str(proc_enb.pid)]
            proc_ps = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
            threads = 0
            thread_ids = []
            while True:
                output = proc_ps.stdout.readline().strip()
                if ('WORKER' in output and not 'TASK' in output):
                    threads += 1
                    thread_id = output.split()[1]
                    thread_ids.append(thread_id)
                    with open('/sys/fs/cgroup/cpuset/user/tasks', 'w') as outfile:
                        subprocess.Popen([loc_exec_echo, str(thread_id)], stdout=outfile)
                
                if (threads == 3):
                    break
            break
        return_code = proc_enb.poll()
        if return_code is not None:
            print('RETURN CODE', return_code)
            # Process has finished, read rest of the output 
            for output in proc_enb.stdout.readlines():
                print(output.strip())
            break
    
    return proc_enb, proc_agent

def start_ue_process(ue_netns = 'ue', tx_port = 2001, rx_port = 2000):
    cmd = ['/home/naposto/tools/srsRAN/bin/srsue']
    cmd.append('/home/naposto/.config/srsran/ue.conf')
    cmd.append('--gw.netns={}'.format(ue_netns))
    cmd.append('--rf.device_name=zmq')
    cmd.append('--rf.device_args=fail_on_disconnect=true,tx_port=tcp://*:{},rx_port=tcp://localhost:{},id=ue,base_srate=23.04e6'.format(tx_port, rx_port))
    proc_ue = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)

    while True:
        output = proc_ue.stdout.readline().strip()
        if ('Network attach successful.' in output):
            subprocess.Popen([loc_exec_ip, 'netns', 'exec', 'ue', loc_exec_ip, 'route', 'add', 'default', 'dev', 'tun_srsue'])
            break
    return proc_ue

def start_ue_iperf_process(ue_netns = 'ue', duration = 300):
    return subprocess.Popen([loc_exec_ip, 'netns', 'exec', ue_netns, 
                             loc_exec_iperf, '-c', iperf_server_address, '-u', '-t', str(duration), ], 
                             stdout=subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)

def kill_process(process, signals_kill_to_send = 1):
    for _ in range(signals_kill_to_send):
        process.send_signal(signal.SIGTERM)

def run_simulation( enb_scheduler, 
                    enb_csv_file,
                    enb_log_file,                   
                    agent_pretrained = False, 
                    agent_seed = 0,
                    agent_results_file = None,
                    agent_pretrained_weights = None,
                    beta_factor = 700):
    proc_iperf_server = None
    proc_enb = None
    proc_agent = None
    proc_ue = None
    proc_epc = None
    proc_iperf_client = None
    

    proc_epc = start_epc_process()
    print('Started srsepc [{}]... '.format(proc_epc.pid))

    proc_enb, proc_agent = start_enb_process(
        beta_factor=beta_factor,
        enb_log_file=enb_log_file,
        csv_report_period=csv_report_period,
        enb_csv_file=enb_csv_file,
        scheduler=enb_scheduler,
        agent_seed=agent_seed,
        agent_episodes=1000,
        agent_results_file=agent_results_file,
        agent_pretrained=agent_pretrained,
        agent_pretrained_weights=agent_pretrained_weights
    )
    if (proc_agent is not None) :
        print('Started scheduler agent [{}]... '.format(proc_agent.pid))    
    print('Started srsenb [{}]... '.format(proc_enb.pid))

    proc_ue  = start_ue_process()
    print('Started srsue1 [{}]... '.format(proc_ue.pid))


    proc_iperf_server = start_iperf_server_process()
    print('Started iperf server [{}]... '.format(proc_iperf_server.pid))

    proc_iperf_client = start_ue_iperf_process()
    print('Started UE1 iperf client [{}]...'.format(proc_iperf_client.pid))
    proc_iperf_client.wait()
    
    print('Killing UE1 iperf client...', end = '')
    kill_process(proc_iperf_client)
    proc_iperf_client = None
    print('Done')

    print('Killing iperf server...', end = '')
    kill_process(proc_iperf_server, signals_kill_to_send = 2)
    proc_iperf_server = None
    print('Done')

    print('Killing srsue1...', end = '')
    kill_process(proc_ue)
    proc_ue = None
    print('Done')

    if (proc_agent is not None):
        print('Killing sceduler agent...', end = '')
        kill_process(proc_agent)
        proc_agent = None
        print('Done')

    print('Killing srsenb...', end = '')
    kill_process(proc_enb)
    proc_enb = None
    print('Done')

    print('Killing srsepc...', end = '')
    kill_process(proc_epc)
    proc_epc = None
    print('Done')

run_simulation(
    enb_scheduler='time_sched_ai',
    enb_csv_file='/home/naposto/phd/nokia/data/csv_46/time_sched_0_enb.csv', 
    enb_log_file='/home/naposto/phd/nokia/data/csv_46/time_sched_0_enb.log',
    agent_results_file='/home/naposto/phd/nokia/data/csv_46/time_sched_0_agent.csv',
    agent_seed=0, 
    agent_pretrained=False,
    agent_pretrained_weights = '/home/naposto/phd/nokia/data/csv_46/train_all.h5'
)