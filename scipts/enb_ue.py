#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Intra Handover Flowgraph
# GNU Radio version: 3.8.1.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import channels
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.qtgui import Range, RangeWidget
from gnuradio import zeromq

from gnuradio import qtgui

BETA_FIFO = '/tmp/beta_fifo'

# gain_levels = [
#         1.0,  .8,  .6, .5,   .4, .35,  .3, .25, .22, .20, 
#         .18, .16, .14, .12, .10, .09, .08, .07, .06, .05]
gain_levels = [
1.0,  .8,  .6, .5,   .4, .35,  .3, .25, .22, .20, 
.18, .16, .14, .12, .10, .08, .06]
gain_level_duration = 30
total_loops = 1

# gain_level_duration = 30
# gain_levels = [
# 1.0,  .8,  .6, .5,   .4, .35,  .3, .28, .27, .26, .25, .24, .23, .22, .21, .20, .19,
# .18, .17, .16, .15, .14, .13, .12, .11, .10, .09, .08, .07, .06]
# gain_levels = [1, 1, 1, 1]
# gain_levels = [1.0, .20, .05]
congestion_levels = [
    0, 200, 400, 600, 800, 1000
                    ]
congestion_levels = [
    0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
                    ]
congestion_levels = [0, 500, 1000]
# congestion_levels = [200]
congestion_level_duration = gain_level_duration * len(gain_levels) * 2

class intra_enb(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Intra Handover Flowgraph")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Channel Tone Response")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)

        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "channel_tone_response")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.noise_level_ue1 = noise_level_ue1 = 0.01
        self.ul_gain_level = self.dl_gain_level = 1
        self.samp_rate = samp_rate = 24e6

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            2048, #size
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            samp_rate, #bw
            '', #name
            1
        )
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)

        labels = ['', '', '', '', '',
            '', '', '', '', '']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ["blue", "red", "green", "black", "cyan",
            "magenta", "yellow", "dark red", "dark green", "dark blue"]
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]

        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_freq_sink_x_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_freq_sink_x_0.set_line_label(i, labels[i])
            self.qtgui_freq_sink_x_0.set_line_width(i, widths[i])
            self.qtgui_freq_sink_x_0.set_line_color(i, colors[i])
            self.qtgui_freq_sink_x_0.set_line_alpha(i, alphas[i])
        import sip
        self._qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_freq_sink_x_0_win)
        


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
        self.connect((self.uplink, 0), (self.qtgui_freq_sink_x_0, 0))
        
        self.connect((self.enb_tx_port, 0), (self.multiplier_dl, 0))
        self.connect((self.multiplier_dl, 0), (self.downlink, 0))
        self.connect((self.downlink, 0), (self.ue_rx_port, 0))
        # self.connect((self.downlink, 0), (self.qtgui_freq_sink_x_0, 0))

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
    
    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "channel_tone_response")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def set_n(self, n):
        self.n = n

import time
import numpy as np

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
                        # tb.set_multiply_level_ue1(gain_levels[len(gain_levels) // 2])
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
                        if (loops == total_loops):
                            break
        except FileNotFoundError as e:
            print('error')
        finally:
            print('Gracefully exiting...')

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
    print('Starting...')
    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        print('Here...')
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    print('Gooo')
    qapp = Qt.QApplication(sys.argv)

    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("Error: failed to enable real-time scheduling.")
    
    tb = top_block_cls()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()    
    
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    # threading.Thread(target=input_thread, args = (tb, )).start()
    # threading.Thread(target=automated_noise_level_setting_thread, args = (tb, )).start()
    # print('aaa')
    # threading.Thread(target=automated_cpu_level_thread).start()
    # t2 = threading.Thread(target=automated_cpu_level_thread())
    # t2 = threading.Thread(target=automated_monitoring_thread(tb,))
    # t2.start()
    
    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()
    # t2.join()


if __name__ == '__main__':
    main()
