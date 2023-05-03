#!/usr/bin/python3

import subprocess, os
import signal
from time import sleep
import os

iperf_target_address = '172.18.0.1'
iperf_duration = '20' # in seconds
spgw_if_address = '172.18.0.1'

proc_grc = None
proc_enb = None
proc_ue = None
proc_epc = None
proc_iperf_client = None
proc_enb_scheduler = None

loc_exec_ip = '/sbin/ip'
loc_exec_iperf = '/usr/bin/iperf'
loc_exec_echo = '/bin/echo'

# import signal

def signal_handler(signal, frame):
    processes = [proc_iperf_client, proc_epc, proc_enb, proc_ue, proc_grc, proc_enb_scheduler]
    for process in processes:
        if process is None:
            continue
        process.kill()
        process.wait()
    exit(2)
    
signal.signal(signal.SIGINT, signal_handler)

def start_srsepc():
    cmd = ['/home/naposto/tools/srsRAN/bin/srsepc']
    cmd.append('/home/naposto/.config/srsran/epc.conf')
    cmd.append('--spgw.sgi_if_addr={}'.format(spgw_if_address))
    cmd.append('--hss.db_file')
    cmd.append('/home/naposto/.config/srsran/user_db.csv')
    proc_epc = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    print('Started srsEPC [{}]... '.format(proc_epc.pid), end='')
    while True:
        output = proc_epc.stdout.readline()
        # print(output)
        if ('SP-GW Initialized' in output):
            print('srsEPC initialized successfully...')
            break
    return proc_epc

def start_channel():
    cmd = ['/usr/bin/python3', '-u', '/home/naposto/phd/nokia/agents/scipts/enb_ue_no_qt.py']
    env = os.environ.copy()
    env['PYTHONPATH'] = '/usr/local/lib/python3/dist-packages:'
    print('environment varable: ' + env['PYTHONPATH'])
    proc_grc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)    
    print('Started GRC [{}]...'. format(proc_grc.pid))
    while True:
        output = proc_grc.stdout.readline()
        # print(output)
        if ('Input threading listening to requests' in output):
            # print('Channel initialized and listening to requests...')
            break
    return proc_grc

def start_srsenb(scheduler):
    cmd = ['/home/naposto/tools/srsRAN/bin/srsenb']
    cmd.append('/home/naposto/.config/srsran/enb.conf')
    cmd.append('--enb_files.sib_config')
    cmd.append('/home/naposto/.config/srsran/sib.conf')
    cmd.append('--enb_files.rr_config')
    cmd.append('/home/naposto/.config/srsran/rr.conf')
    cmd.append('--enb_files.rb_config')
    cmd.append('/home/naposto/.config/srsran/rb.conf')
    sched_string = 'time_sched_ai'
    if (scheduler == 'srs'):
        sched_string = 'time_rr'        
    cmd.append('--scheduler.policy={}'.format(sched_string))
    cmd.append('--scheduler.ul_snr_avg_alpha=.5')
    cmd.append('--expert.pusch_beta_factor=0')
    cmd.append('--log.phy_level=warning')
    cmd.append('--log.filename=/tmp/enb.log')
    cmd.append('--rf.device_name=zmq')
    cmd.append('--rf.device_args=fail_on_disconnect=true,tx_port=tcp://*:2101,rx_port=tcp://localhost:2100,id=enb,base_srate=23.04e6')
    proc_enb = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    print('Started srsENB [{}]... '.format(proc_enb.pid), end='')
    while True:
        output = proc_enb.stdout.readline()
        # print(output)
        if ('Built in Release mode using commit' in output):
            print('srsENB initialized...')
            break
    return proc_enb

def start_srsue():
    cmd = ['/home/naposto/tools/srsRAN/bin/srsue']
    cmd.append('/home/naposto/.config/srsran/ue.conf')
    cmd.append('--gw.netns=ue')
    cmd.append('--rf.device_name=zmq')
    cmd.append('--rf.device_args=fail_on_disconnect=true,tx_port=tcp://*:2001,rx_port=tcp://localhost:2000,id=ue,base_srate=23.04e6')
    proc_ue = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)
    print('Started srsUE [{}]... '.format(proc_ue.pid), end = '')
    while True:
        output = proc_ue.stdout.readline()
        # print(output)
        if ('Attaching UE..' in output):
            print('srsUE initialized...')
            break
    return proc_ue

def start_srsenb_scheduler(scheduler, path_results):
    cmd = ['/usr/bin/python3', '/home/naposto/phd/nokia/agents/pipe_in.py']
    cmd.append('-m')
    if (scheduler in ['athena', 'athena_mcs']):
        cmd.append('athena')
        cmd.append('--actions')
        if (scheduler == 'athena'):
            cmd.append('2')
        else:
            cmd.append('1')
    else:
        cmd.append('srs')
    cmd.append('--results')
    cmd.append(path_results)
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/naposto/.local/lib/python3.8/site-packages:'
    proc_srsenb_scheduler = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, universal_newlines=True) 
    # proc_srsenb_scheduler = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE) 
    print('Started srsENB scheduler [{}]...'. format(proc_srsenb_scheduler.pid))
    while True:
        output = proc_srsenb_scheduler.stdout.readline()
        # print(output)
        if ('Waiting working threads' in output):
        # if ('Worker threads started successfully' in output):
            print('srsENB scheduler initialized...')
            break
    return proc_srsenb_scheduler

def echo_to_cpuset():
    cmd = ['/usr/local/bin/echo_to_cpuset']
    echo_to_cpuset = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    print('echo_to_cpuset script called [{}]'.format(echo_to_cpuset.pid))
    
def ip_route_add_default():
    proc = subprocess.Popen([loc_exec_ip, 'netns', 'exec', 'ue', loc_exec_ip, 'route', 'add', 'default', 'dev', 'tun_srsue'])
    print('ip route add default called [{}]'.format(proc.pid))
    streamdata = proc.communicate()[0]
    return_code = proc.returncode
    if (return_code != 0):
        print('proc exited with error code {}'.format(return_code))
    return

def iperf_ue():
    proc_iperf_client = subprocess.Popen([loc_exec_ip, 'netns', 'exec', 'ue', loc_exec_iperf, '-c', iperf_target_address, '-u', '-t', iperf_duration, '-b', '20G'], stdout=subprocess.PIPE, stderr = subprocess.PIPE, universal_newlines=True)
    print('iperf called [{}]'.format(proc_iperf_client.pid))
    streamdata = proc_iperf_client.communicate()[0]
    return proc_iperf_client

def set_channel_beta_gain(channel_process, beta, gain):
    # channel_process.stdin.write('beta={},gain={}\n'.format(beta, gain).encode("utf-8"))
    channel_process.stdin.write('beta={},gain={}\n'.format(beta, gain))
    channel_process.stdin.flush()


def run_simulation(gain=1, beta=0, scheduler='athena', results_file_path=None):
    try:
        proc_epc = start_srsepc()
        proc_enb = start_srsenb(scheduler)
        proc_ue  = start_srsue()
        proc_enb_scheduler = start_srsenb_scheduler(scheduler, results_file_path)
        proc_grc = start_channel()
        while True:
            output = proc_ue.stdout.readline().strip()
            if ('Network attach successful.' in output):
                break
        echo_to_cpuset()            
        ip_route_add_default()
        set_channel_beta_gain(proc_grc, beta, gain)  
        proc_iperf_client = iperf_ue()
        
        print('Stopping iperf client...', end='')
        proc_iperf_client.kill()
        proc_iperf_client.wait()
        proc_iperf_client = None
        print('Done')

        print('Stopping srsUE...', end='')
        proc_ue.kill()
        proc_ue.wait()
        proc_ue = None
        print('Done')

        print('Stopping srsENB...', end='')
        proc_enb.kill()
        proc_enb.wait()
        proc_enb = None
        print('Done')

        print('Stopping GRC channel...', end='')
        proc_grc.kill()
        proc_grc.wait()
        proc_grc = None
        print('Done')

        print('Killing srsEPC...', end='')
        proc_epc.kill()
        proc_epc.wait()
        proc_epc = None
        print('Done')

        print('Killing srsENB Scheduler...', end='')
        proc_enb_scheduler.terminate()
        proc_enb_scheduler.wait()
        proc_enb_scheduler = None
        print('Done')
    except Exception as e:
        signal_handler(None, None)




arr_beta = [0, 500, 1000]

dict_gain_to_snr = {
    1: 30,
    .9: 29,
    .8: 28,
    .7: 27,
    .65: 25,
    .6: 25,
    .55: 25,
    .50: 24,
    .45: 23,
    .40: 22,
    .38: 21,
    .35: 20,
    .32: 20,
    .30: 19,
    .28: 19,
    .25: 18,
    .23: 17,
    .20: 16,
    .18: 15,
    .165: 14,
    .15: 13,
    .14: 13,
    .13: 12,
    .12: 12,
    .11: 11,
    .10: 10,
    .09: 9,
    .08: 8,
    .07: 8,
    .065: 7,
    .06: 6,
    .058: 6,
    .055: 6,
    .05: 5
}

parent_folder = '/home/naposto/phd/nokia/ddpg_new/data/'
for algorithm in ['srs']:
    for beta in [0, 500, 1000]:
        for gain in list(dict_gain_to_snr.keys()):
            results_file_suffix = '{}_{}_{}.csv'.format(algorithm, beta, gain)
            results_file = parent_folder + results_file_suffix
            run_simulation(gain=gain,beta=beta,scheduler=algorithm,results_file_path=results_file)