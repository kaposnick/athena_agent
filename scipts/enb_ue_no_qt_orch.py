#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Intra Handover Flowgraph
# GNU Radio version: 3.8.1.0

from gnuradio import channels
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import zeromq
import time
import numpy as np
np.random.seed(0)
import threading
import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(0)
from tensorflow.keras import layers


BETA_TO_SRS = '/tmp/beta_fifo'
BETA_FROM_SRS = '/tmp/beta_from_srs'

gain_levels = [
1.0,  .8,  .6, .5,   .4, .35,  .3, .25, .22, .20, 
.18, .16, .14, .12, .10, .08, .06]
gain_levels = [
    .5,   .4, .35,  .3, .25, .22, .20, 
    .18, .16, .14, .12
]

gain_levels = [
    .5, .5, .5, .12,.12,.12, .5, .5, .5
]

gain_level_duration = 3
total_loops = 2

congestion_levels = [
    0, 200, 400, 600, 800, 1000
                    ]
congestion_levels = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
                    ]
congestion_levels = np.array(congestion_levels, dtype=np.float32).reshape((-1,1))

cpu_operating = 0
gain_operating = 1.0

window_length = 100

target_throughput = 13
time_window       = 3

percentile        = 1
func_percentile   = [np.percentile, [percentile]]
func_mean         = [np.mean      , []]
snr_calculation   = func_percentile


class agent():
    def __init__(self, context_dim, action_dim_actor, action_dim_critic) -> None:
        self.context_dim = context_dim
        self.action_dim_actor = action_dim_actor
        self.action_dim_critic = action_dim_critic
        self.actor = self.get_actor()
        self.critic = self.get_critic()
        self.context_min = np.array([0, 18], dtype=np.float32)
        self.context_max = np.array([1000, 49], dtype=np.float32)
        self.mcs_prb_min = np.array([0, 1], dtype=np.float32)
        self.mcs_prb_max = np.array([24, 45], dtype=np.float32)
        self.build_mcs_tbs_arrays()

    def to_tbs(self, mcs, prb):
        tbs = 0
        if (prb > 0):
            i_tbs = self.I_MCS_TO_I_TBS[mcs]
            tbs = self.tbs_table[i_tbs][prb - 1]
        return tbs

    def build_mcs_tbs_arrays(self):
        PROHIBITED_COMBOS = [(0, 0), (0, 1), (0,2), (0, 3), 
                  (1, 0), (1, 1), (1, 2),
                  (2, 0), (2, 1),
                  (3, 0), 
                  (4, 0), 
                  (5, 0), 
                  (6, 0)]
        PRB_SPACE = np.array(
                    [1, 2, 3, 4, 5, 6, 8, 9, 
                      10, 12, 15, 16, 18, 
                      20, 24, 25, 27, 
                      30, 32, 36, 40, 45], dtype = np.float16)
        MCS_SPACE = np.arange(0, 25+1, dtype=np.float16)
        self.I_MCS_TO_I_TBS = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
             10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
             19, 20, 21, 22, 23, 24, 25, 26]
        )
        import json
        tbs_path = '/home/naposto/phd/generate_lte_tbs_table/samples/cpp_tbs.json'
        with open(tbs_path) as tbs_json:
            self.tbs_table = json.load(tbs_json)
        mapping_array = []
        for mcs in MCS_SPACE:
            for prb in PRB_SPACE:
                combo = ( self.I_MCS_TO_I_TBS[int(mcs)], int(prb) - 1)
                if combo in PROHIBITED_COMBOS:
                    continue
                mapping_array.append(
                    {   
                    'tbs': self.to_tbs(int(mcs), int(prb)),
                    'mcs': mcs,
                    'prb': prb
                    }
                )
        self.mapping_array = sorted(mapping_array, key = lambda el: (el['tbs'], el['mcs']))
        self.mcs_prb_array = np.array([np.array([x['mcs'], x['prb']]) for x in self.mapping_array]) # sort by tbs/mcs

    def normalize(self, input, min, max):
        return (input - min) / (max - min)
    
    def denormalize(self, input, min, max):
        return (max-min) * input + min

    def get_actor(self):
        state_input = keras.Input(shape = (self.context_dim))
        x = layers.Dense(16, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (state_input)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        norm_params = layers.Dense(self.action_dim_actor, kernel_initializer = keras.initializers.HeNormal(), activation='sigmoid')(x)
        actor = keras.Model(state_input, norm_params)
        return actor

    def get_critic(self):
        state_input = keras.Input(shape = (self.context_dim))
        action_input = keras.Input(shape = (self.action_dim_critic))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(16, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(256, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        x = layers.Dense(128, activation = 'relu', kernel_initializer = keras.initializers.HeNormal()) (x)
        q = layers.Dense(1, kernel_initializer = keras.initializers.HeNormal()) (x)
        critic = keras.Model(inputs = [state_input, action_input], outputs=q)
        return critic
    
    def load_actor(self, path):
        self.actor.load_weights(path)

    def load_critic(self, path):
        self.critic.load_weights(path)

    def evaluate(self, context, k=9):
        context            = self.normalize(context, self.context_min, self.context_max)
        mcs_prb_normalized = self.actor.predict(context, verbose=0)[0]
        mcs_prb            = self.denormalize(mcs_prb_normalized, self.mcs_prb_min, self.mcs_prb_max)
        l2_norms           = np.linalg.norm(mcs_prb - self.mcs_prb_array, axis=1)
        partition          = np.argpartition(l2_norms, k)
        k_closest_mcs_prb  = self.mcs_prb_array[partition[:k]]

        k_closest_mcs_prb_normalized = self.normalize(k_closest_mcs_prb, self.mcs_prb_min, self.mcs_prb_max)
        context_extended   = np.broadcast_to(context, (k, context.shape[1]))
        q_values           = self.critic([context_extended, k_closest_mcs_prb_normalized])
        argmin_q_value     = np.argmax(q_values)
        mcs, prb           = k_closest_mcs_prb[argmin_q_value]
        return int(mcs), int(prb), self.to_tbs(int(mcs), int(prb)) / (1024**2) * 1000, *mcs_prb


class intra_enb(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Intra Handover Flowgraph")

        ##################################################
        # Variables
        ##################################################
        self.noise_level_ue1 = noise_level_ue1 = 0.01
        self.ul_gain_level = self.dl_gain_level = 1        

        ##################################################
        # Blocks
        ##################################################
        self.enb_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2101', 100, False, -1)
        self.ue_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2001', 100, False, -1)
        self.ue_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2000', 100, False, -1)
        self.enb_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2100', 100, False, -1)
        self.multiplier_ul = blocks.multiply_const_cc(self.ul_gain_level)
        self.multiplier_dl = blocks.multiply_const_cc(self.dl_gain_level)
        self.uplink = channels.awgn_model(
            noise_voltage=noise_level_ue1,
            noise_seed=0)
        self.uplink.set_block_alias("UE1 UPLINK")
        self.downlink = channels.awgn_model(
            noise_voltage=noise_level_ue1,
            noise_seed=0)
        self.downlink.set_block_alias("UE1 DOWNLINK")



        ##################################################
        # Connections
        ##################################################
        self.connect((self.ue_tx_port, 0), (self.multiplier_ul, 0))
        self.connect((self.multiplier_ul, 0), (self.uplink, 0))
        self.connect((self.uplink, 0), (self.enb_rx_port, 0))
        
        self.connect((self.enb_tx_port, 0), (self.multiplier_dl, 0))
        self.connect((self.multiplier_dl, 0), (self.downlink, 0))
        self.connect((self.downlink, 0), (self.ue_rx_port, 0))

    def get_noise_level_ue1(self):
        return self.noise_level_ue1

    def set_noise_level_ue1(self, noise_level_ue1):
        # print('Changing noise voltage to: {}'.format(noise_level_ue1))
        self.noise_level_ue1 = noise_level_ue1
        self.uplink.set_noise_voltage(self.noise_level_ue1)
        self.downlink.set_noise_voltage(self.noise_level_ue1)

    def set_multiply_level_ue1(self, multiply_level_ue1):
        self.ul_gain_level = multiply_level_ue1
        self.dl_gain_level = multiply_level_ue1
        self.multiplier_ul.set_k(self.ul_gain_level)
        self.multiplier_dl.set_k(self.dl_gain_level)

def cost_model_linear(x, beta_min=0, cost_min=100, beta_max=1000, cost_max=1):
    x1, y1 = beta_min, cost_min
    x2, y2 = beta_max, cost_max
    alpha = (y2-y1)/(x2-x1)
    beta  = (y1+y2)/2 - alpha/2*(x1+x2)
    return alpha * x + beta
costs_per_cpu = cost_model_linear(congestion_levels).reshape((-1,))

def predict_throughput(model, context):
    _len = context.shape[0]
    np_thr = np.zeros(shape=(_len))
    for i in np.arange(_len):
        context_i = context[i,:].reshape((1, context.shape[1]))
        _, _, thr, _, _ = model.evaluate(context_i)
        np_thr[i] = thr
    return np_thr

def recalculate_cpu(window, target_throughput, model):
    target_snr  = snr_calculation[0](window, *snr_calculation[1])
    snr_column = np.full_like(congestion_levels, fill_value=target_snr)
    context = np.hstack([congestion_levels, snr_column])
    throughput = predict_throughput(model, context)

    valid_indeces = np.where(throughput > target_throughput)
    valid_costs_per_cpu = costs_per_cpu[valid_indeces]
    if (len(valid_costs_per_cpu) > 0):
        costs_valid = np.argmin(costs_per_cpu[valid_indeces])
        cpu  = congestion_levels[costs_valid][0]
    else:
        objective = np.abs(throughput - target_throughput)
        costs_valid = np.argmin(objective)
        cpu = congestion_levels[costs_valid][0]
    return cpu

import queue
from queue import Queue

stop = False
threads_ready_counter = 0
threads_ready_lock = threading.Lock()

def wait_for_threads():
    with threads_ready_lock:
        global threads_ready_counter
        threads_ready_counter += 1
        print('Ready {}/{}'.format(threads_ready_counter, 3))    

    while (threads_ready_counter != 3):
        time.sleep(1)

def orchestrator_trigger(_queue, lock, fd_write):    
    global stop
    model = agent(context_dim=2, action_dim_actor=2, action_dim_critic=2)
    model.load_actor('/home/naposto/phd/nokia/agents/model/ddpg_actor_99.h5')
    model.load_critic('/home/naposto/phd/nokia/agents/model/ddpg_critic_99.h5')
    global cpu_operating
    window = []
    wait_for_threads()
    while(True):
        time.sleep(time_window)
        with lock:
            try:
                while(True):
                    item = _queue.get(block=False)
                    window.append(item)         
            except queue.Empty:
                pass
        
        if (stop):
            break
        if (len(window)) == 0:
            print('Empty window')
            continue
        snrs = np.array(window)
        cpu_operating = recalculate_cpu(snrs, target_throughput=target_throughput, model=model)
        current_gain_level_bytes = (int(gain_operating* 1000)).to_bytes(2, byteorder='little')
        congestion_level_bytes = int(cpu_operating).to_bytes(4, byteorder='little')                        
        print('=> Orchestrator: {}'.format(cpu_operating))
        fd_write.write(congestion_level_bytes + current_gain_level_bytes)
        fd_write.flush()
        window.clear()
    print('Orch: exiting')
                

def thread_fn_snr_reader(fd_read, fd_write):
    global stop
    trigger_thread = None
    _queue  = Queue()
    lock   = threading.Lock()
    trigger_thread = threading.Thread(target=orchestrator_trigger, args=(_queue, lock, fd_write))
    trigger_thread.start()
    wait_for_threads()
    while(True):
        content = fd_read.read(8)
        if (len(content) < 0):
            print('EOF')
            break
        tti = int.from_bytes(content[:4], "little")
        snr = int.from_bytes(content[4:], "little") / 1000
        with lock:
            _queue.put(snr)
        if (stop):
            break
    print('SNR Reader: received exit signal')
    if (trigger_thread is not None):
        print('SNR Reader: waiting for orch to finish')
        trigger_thread.join()
        print('SNR Reader: orch finished')
    print('SNR Reader: exiting')
    



def thread_fn_gain_adjuster(tb, fd_writer):        
    global stop
    global gain_operating
    print('Opening beta fifo socket successful. Going to sleep for 10 seconds')                
    time.sleep(10) # this sleep is necessary as the echo_to_cpuset and the iperf startup scripts are initiated    
    loops = 0
    wait_for_threads()
    print('Starting simulation...')                
    while (True):
        print('Loop {}/{}'.format(loops + 1, total_loops))
        current_gain_level_idx = 0
        direction = 1
        hit_low_snr_first_time = True
        while (1):
            gain_operating = gain_levels[current_gain_level_idx]
            tb.set_multiply_level_ue1(gain_operating)
            current_gain_level_bytes = (int(gain_operating* 1000)).to_bytes(2, byteorder='little')
            congestion_level_bytes = int(cpu_operating).to_bytes(4, byteorder='little')                        
            print('==> Gain Adjuster: {}'.format(gain_operating))
            fd_writer.write(congestion_level_bytes + current_gain_level_bytes)
            fd_writer.flush()
            time.sleep(gain_level_duration)
            if (stop):
                break
            if (current_gain_level_idx + direction == len(gain_levels) or
                current_gain_level_idx + direction == -1):
                if (current_gain_level_idx == len(gain_levels) - 1 and hit_low_snr_first_time):
                    hit_low_snr_first_time = False
                    continue
                elif (current_gain_level_idx == 0):
                    break
                direction = -direction                        
            current_gain_level_idx += direction
        loops += 1
        if (loops == total_loops):
            break
    print('Gain Adjuster: Exiting...')
    stop=True




def main(top_block_cls=intra_enb, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")
    tb = top_block_cls()
    tb.start()

    t1 = None
    t2 = None

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        global stop
        stop=True
        if (t1 is not None):
            t1.join()
        sys.exit(0)
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    
    try:
        with open(BETA_FROM_SRS, mode='rb') as fd_read, open(BETA_TO_SRS, mode='wb') as fd_write:
            t1 = threading.Thread(target=thread_fn_gain_adjuster, args=(tb, fd_write))    
            t2 = threading.Thread(target=thread_fn_snr_reader, args=(fd_read, fd_write))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
    except IOError as e:
        print('Error')


    

def main2():
    window = np.random.normal(20, 1, size=(window_length))
    target_throughput = 10
    model = agent(context_dim=2, action_dim_actor=2, action_dim_critic=2)
    model.load_actor('/home/naposto/phd/nokia/agents/model/ddpg_actor_99.h5')
    model.load_critic('/home/naposto/phd/nokia/agents/model/ddpg_critic_99.h5')
    recalculate_cpu(window, target_throughput, model)
    pass

if __name__ == '__main__':
    main()
