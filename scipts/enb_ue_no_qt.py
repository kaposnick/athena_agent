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

BETA_FIFO = '/tmp/beta_fifo'

gain_levels = [
        1.0,  .8,  .6, .5,   .4, .35,  .3, .25, .22, .20, 
        .18, .16, .14, .12, .10, .09, .08, .07, .06, .05]
gain_level_duration = 3
# gain_levels = [1.0, .20, .05]
congestion_levels = [
    0, 200, 400, 600, 800, 1000
                    ]
# congestion_levels = [200]
congestion_level_duration = gain_level_duration * len(gain_levels) * 2

class intra_enb(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Intra Handover Flowgraph")

        ##################################################
        # Variables
        ##################################################
        self.noise_level_ue1 = noise_level_ue1 = 0.01
        self.multiply_level_ue1 = multiple_level_ue1 = 1

        ##################################################
        # Blocks
        ##################################################
        self.enb_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2101', 100, False, -1)
        self.ue_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, 'tcp://localhost:2001', 100, False, -1)
        self.ue_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2000', 100, False, -1)
        self.enb_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, 'tcp://*:2100', 100, False, -1)
        self.multiplier = blocks.multiply_const_cc(multiple_level_ue1)
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
        self.connect((self.ue_tx_port, 0), (self.multiplier, 0))
        self.connect((self.multiplier, 0), (self.uplink, 0))
        self.connect((self.uplink, 0), (self.enb_rx_port, 0))
        
        self.connect((self.enb_tx_port, 0), (self.downlink, 0))
        self.connect((self.downlink, 0), (self.ue_rx_port, 0))

    def get_noise_level_ue1(self):
        return self.noise_level_ue1

    def set_noise_level_ue1(self, noise_level_ue1):
        # print('Changing noise voltage to: {}'.format(noise_level_ue1))
        self.noise_level_ue1 = noise_level_ue1
        self.uplink.set_noise_voltage(self.noise_level_ue1)
        self.downlink.set_noise_voltage(self.noise_level_ue1)

    def set_multiply_level_ue1(self, multiply_level_ue1):
        # print('Changing multiply level ue1 to {}'.format(multiply_level_ue1))
        self.multiply_level_ue1 = multiply_level_ue1
        self.multiplier.set_k(self.multiply_level_ue1)

import time
import numpy as np

def automated_monitoring_thread(tb):    
    current_cpu_level_idx = 0
    
    is_file_open = False
    while (not is_file_open):
        try:
            with open(BETA_FIFO, mode = 'wb') as file_write:
                print('Opening beta fifo socket successful. Going to sleep')                
                time.sleep(10) # this sleep is necessary as the echo_to_cpuset and the iperf startup scripts are initiated    
                while (True):
                    # iterate through the CPU congestion levels (outer loop - slower)
                    congestion_level = congestion_levels[current_cpu_level_idx]
                    congestion_level_bytes = congestion_level.to_bytes(4, byteorder='little')
                    file_write.write(congestion_level_bytes)
                    file_write.flush()                    
                    print('Setting CPU Congestion level {}'.format(congestion_level))

                    current_gain_level_idx = 0
                    direction = 1
                    hit_low_snr_first_time = True
                    while (1):
                        tb.set_multiply_level_ue1(gain_levels[current_gain_level_idx])
                        print('Setting UE Gain level {}'.format(gain_levels[current_gain_level_idx]))
                        time.sleep(gain_level_duration)
                        if (current_gain_level_idx + direction == len(gain_levels) or
                            current_gain_level_idx + direction == -1):
                            if (current_gain_level_idx == len(gain_levels) - 1 and hit_low_snr_first_time):
                                hit_low_snr_first_time = False
                                continue
                            elif (current_gain_level_idx == 0):
                                break
                            direction = -direction
                        
                        current_gain_level_idx += direction
                    
                    current_cpu_level_idx += 1
                    current_cpu_level_idx %= len(congestion_levels)
        except FileNotFoundError as e:
            print('error')
            pass

def automated_cpu_level_thread():
    print('Started CPU occupancy thread...', end='')
    
    print('Done')
    
    current_level_idx = 0
    is_file_open = False
    while (not is_file_open):
        try:
            with open(BETA_FIFO, mode = 'wb') as file_write:
                print('Opening beta fifo socket...')
                time.sleep(20)
                while (True):
                    congestion_level = congestion_levels[current_level_idx]
                    congestion_level_bytes = congestion_level.to_bytes(4, byteorder='little')
                    file_write.write(congestion_level_bytes)
                    file_write.flush()                    
                    print('Setting CPU Congestion level {}'.format(congestion_level))
                    time.sleep(congestion_level_duration)
                    current_level_idx += 1
                    current_level_idx %= len(congestion_levels)
        except FileNotFoundError as e:
            print('error')
            pass

def automated_noise_level_setting_thread(tb):
    print('Starting Gain level thread...', end = '')
    time.sleep(20)
    print('Done')
    direction = 1
    current_level_idx = 0
    while(True):
        tb.set_multiply_level_ue1(gain_levels[current_level_idx])
        print('Setting UE Gain level {}'.format(gain_levels[current_level_idx]))
        time.sleep(gain_level_duration)
        if (direction == 1):
            if current_level_idx + 1 < len(gain_levels):
                current_level_idx += 1
            else:
                current_level_idx -=1
                direction = -1
                # current_level_idx = 0
        else:
            if current_level_idx - 1 >= 0:
                current_level_idx -= 1
            else:
                current_level_idx += 1
                direction = 1

        if (current_level_idx < 0):
            current_level_idx = 0
        if (current_level_idx == len(gain_levels)):
            current_level_idx = len(gain_levels) - 1


def input_thread(tb):
    for line in sys.stdin:
        try:
            print('Received: {}'.format(line))
            line = line.strip()
            noise_levels = line.split()
            # tb.set_noise_level_ue1(float(noise_levels[0]))
            tb.set_multiply_level_ue1(float(noise_levels[0]))
        except Exception as e:
            print('Exception occurred: {}'.format(e))


import threading

def main(top_block_cls=intra_enb, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()    
    # threading.Thread(target=input_thread, args = (tb, )).start()
    # threading.Thread(target=automated_noise_level_setting_thread, args = (tb, )).start()
    # print('aaa')
    # threading.Thread(target=automated_cpu_level_thread).start()
    # t2 = threading.Thread(target=automated_cpu_level_thread())
    t2 = threading.Thread(target=automated_monitoring_thread(tb,))
    
    t2.start()
    # t2.join()
    tb.wait()


if __name__ == '__main__':
    main()
