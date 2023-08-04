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
from gnuradio import gr
from gnuradio import zeromq
import sys
import signal
import threading
import argparse

BETA_FIFO = '/tmp/beta_fifo'
gain_levels = [
1.0,  .8,  .6, .5,   .4, .35,  .3, .25, .22, .20, 
.18, .16, .14, .12, .10, .08, .06]
gain_level_duration = 30

# gain_levels = [
# 1.0,  .8,  .6, .5,   .4, .35,  .3, .28, .27, .26, .25, .24, .23, .22, .21, .20, .19,
# .18, .17, .16, .15, .14, .13, .12, .11, .10, .09, .08, .07, .06]
congestion_levels = [0, 500, 1000]
total_loops = 1

class athena_wireless_channel(gr.top_block):

    def __init__(self, enb_tx, enb_rx, ue_tx, ue_rx):
        gr.top_block.__init__(self, "ATHENA WIRELESS CHANNEL")

        ##################################################
        # Variables
        ##################################################
        self.noise_level_ue1 = noise_level_ue1 = 0.01
        self.ul_gain_level = self.dl_gain_level = 1        

        ##################################################
        # Blocks
        ##################################################
        # enb_tx = 'tcp://localhost:2101'
        # ue_tx  = 'tcp://localhost:2001'
        # ue_rx  = 'tcp://*:2000'
        # enb_rx = 'tcp://*:2100'
        
        self.enb_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, enb_tx, 100, False, -1)
        self.ue_tx_port = zeromq.req_source(gr.sizeof_gr_complex, 1, ue_tx, 100, False, -1)
        self.ue_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, ue_rx, 100, False, -1)
        self.enb_rx_port = zeromq.rep_sink(gr.sizeof_gr_complex, 1, enb_rx, 100, False, -1)
        self.multiplier_ul = blocks.multiply_const_cc(self.ul_gain_level)
        self.multiplier_dl = blocks.multiply_const_cc(self.dl_gain_level)
        self.uplink = channels.channel_model(
            noise_voltage=noise_level_ue1,
            noise_seed=0)
        self.uplink.set_block_alias("UE1 UPLINK")
        self.downlink = channels.channel_model(
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

    def set_multiply_level_ue1(self, multiply_level_ue1):
        self.ul_gain_level = multiply_level_ue1
        self.dl_gain_level = multiply_level_ue1
        self.multiplier_ul.set_k(self.ul_gain_level)
        self.multiplier_dl.set_k(self.dl_gain_level)

import time

def parse_input_token(token, prefix, sep='='):
    sub_tokens = token.split(sep)
    if (len(sub_tokens) != 2):
        return None
    if (sub_tokens[0] != prefix):
        return None
    return float(sub_tokens[1])

def input_thread(tb):
    is_file_open = False
    while (not is_file_open):
        try:
            with open(BETA_FIFO, mode='wb') as file_write:
                is_file_open = True
                print('Input threading listening to requests...')
                for line in sys.stdin:
                    try:
                        line = line.strip()
                        print('Received: {}'.format(line))
                        tokens = line.split(',')
                        if (len(tokens) != 2):
                            continue
                        tok_beta, tok_gain = tokens
                        beta = parse_input_token(tok_beta, 'beta')
                        gain = parse_input_token(tok_gain, 'gain')
                        if (beta == None or gain == None):
                            continue                        
                        
                        tb.set_multiply_level_ue1(gain)
                        current_gain_level_bytes = (int(gain * 1000)).to_bytes(2, byteorder='little')
                        congestion_level_bytes   = (int(beta)).to_bytes(4, byteorder='little')
                        file_write.write(congestion_level_bytes + current_gain_level_bytes)
                        file_write.flush()
                    except Exception as e:
                        print('Exception occurred: {}'.format(e))
        except FileNotFoundError as e:
            print('error')
        finally:
            print('Grafcefully exiting...')
    

def automated_monitoring_thread(tb):    
    current_cpu_level_idx = 0
    
    is_file_open = False
    while (not is_file_open):
        try:
            with open(BETA_FIFO, mode = 'wb') as file_write:
                is_file_open = True
                print('Opening beta fifo socket successful. Going to sleep')                
                time.sleep(10) # this sleep is necessary as the echo_to_cpuset and the iperf startup scripts are initiated    
                loops = 0
                while (True):
                    if (current_cpu_level_idx == 0):
                        print('Loop {}/{}'.format(loops+1, total_loops))
                    # iterate through the CPU congestion levels (outer loop - slower)
                    congestion_level = congestion_levels[current_cpu_level_idx]
                    congestion_level_bytes = congestion_level.to_bytes(4, byteorder='little')
                    gain_level_byte = int(0).to_bytes(2, byteorder='little')
                    file_write.write(congestion_level_bytes + gain_level_byte)
                    file_write.flush()                    
                    print('Setting CPU Congestion level {}'.format(congestion_level))

                    current_gain_level_idx = 0
                    direction = 1
                    hit_low_snr_first_time = True
                    while (1):
                        tb.set_multiply_level_ue1(gain_levels[current_gain_level_idx])
                        current_gain_level_bytes = (int(gain_levels[current_gain_level_idx]* 1000)).to_bytes(2, byteorder='little')
                        file_write.write(congestion_level_bytes + current_gain_level_bytes)
                        file_write.flush()
                        # print('Setting UE Gain level {}'.format(gain_levels[current_gain_level_idx]))
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
                    # break
                    current_cpu_level_idx += 1
                    current_cpu_level_idx %= len(congestion_levels)
                    if (current_cpu_level_idx == 0):
                        loops += 1
        except FileNotFoundError as e:
            print('error')
        finally:
            print('Gracefully exiting...')



def init_parser():
    parser = argparse.ArgumentParser(description="Python process for setting beta and channel gain")
    parser.add_argument('-m', '--mode', choices=['cmd', 'loop'], default='cmd', dest='mode', help='Mode of operation: Loop through beta and gain or from command line')
    parser.add_argument('--ue_tx', dest='ue_tx')
    parser.add_argument('--ue_rx', dest='ue_rx')
    parser.add_argument('--enb_tx', dest='enb_tx')
    parser.add_argument('--enb_rx', dest='enb_rx')

    return parser

def main(top_block_cls=athena_wireless_channel, options=None):
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")
    parser = init_parser()
    args = parser.parse_args()
    
    mode = args.mode
    ue_tx = args.ue_tx
    ue_rx = args.ue_rx
    enb_tx = args.enb_tx
    enb_rx = args.enb_rx

    tb = top_block_cls(enb_tx, enb_rx, ue_tx, ue_rx)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    if (mode == 'loop'):
        print('Loop mode')
        t2 = threading.Thread(target=automated_monitoring_thread, args=(tb, ))
    elif (mode == 'cmd'):
        print('Input mode')
        t2 = threading.Thread(target=input_thread, args=(tb,))
    
    t2.start()
    t2.join()
    tb.wait()


if __name__ == '__main__':
    main()
