import multiprocessing as mp
import queue
from common_utils import MODE_SCHEDULING_RANDOM

class LogProcess(mp.Process):
    '''
        This process is responsible for logging the scheduling samples.
        There are two scheduling samples:
        1) Data returned by a scheduler running infinitely that will be further processed for training
        2) Data returned by the ATHENA/srsRAN scheduler that are used to evaluate the solutions
    '''

    def __init__(self, log_queue: mp.Queue, scheduling_mode, log_file, stop_flag: mp.Value):
        super(LogProcess, self).__init__()
        self.log_queue = log_queue
        self.scheduling_mode = scheduling_mode
        self.log_file = log_file
        self.stop_flag = stop_flag

    def run(self):
        import signal
        signal.signal(signal.SIGINT , self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        if (self.scheduling_mode == MODE_SCHEDULING_RANDOM):
            self.sched_mode_random()
        else:
            self.sched_mode_inference()
        print('Log thread exiting...')

    def exit_gracefully(self, signum, frame):
        self.stop_flag.value = 1

    def sched_mode_random(self):
        with open(self.log_file, 'w') as file:
            file.write('|'.join(['cpu', 'snr', 'mcs', 'prb', 'crc', 'decoding_time', 'snr_decode', 'noise_decode', 'snr_decode_custom', 'gain']) + '\n')
            sample_idx = 0
            while self.stop_flag.value == 0:
                try:
                    sample = self.log_queue.get(block = True, timeout = 2)
                    timestamp = sample['timestamp']
                    tti       = sample['tti']
                    hrq       = sample['hrq']
                    mcs       = sample['mcs']
                    prb       = sample['prb']
                    tbs       = sample['tbs']
                    crc       = sample['crc']
                    dec_time  = sample['dec_time']
                    cpu       = sample['cpu']
                    snr       = sample['snr']
                    gain      = sample['gain']
                    snr_decode       = sample['snr_decode']
                    noise_decode       = sample['noise_decode']
                    snr_custom       = sample['snr_custom']

                    fields = [cpu, snr, mcs, prb, crc, dec_time, snr_decode, noise_decode, snr_custom, gain]
                    record = [str(x) for x in fields]
                    
                    file.write('|'.join(record) + '\n')
                    sample_idx += 1
                    if (sample_idx == 10):
                        file.flush()
                        sample_idx = 0
                except queue.Empty:
                    pass

    def sched_mode_inference(self):
        columns = [
                'timestamp', 'tti', 'hrq', 
                'mcs', 'prb', 'tbs', 
                'crc' , 'dec_time', 
                'cpu', 'snr', 'gain', 'snr_decode', 'noise_decode', 'snr_decode_custom']
        with open(self.log_file, 'w') as file:
            file.write('|'.join(columns) + '\n')
            sample_idx = 0
            while self.stop_flag.value == 0:
                try:
                    sample = self.log_queue.get(block = True, timeout = 2)
                    timestamp = sample['timestamp']
                    tti       = sample['tti']
                    hrq       = sample['hrq']
                    mcs       = sample['mcs']
                    prb       = sample['prb']
                    tbs       = sample['tbs']
                    crc       = sample['crc']
                    dec_time  = sample['dec_time']
                    cpu       = sample['cpu']
                    snr       = sample['snr']
                    gain      = sample['gain']
                    snr_decode       = sample['snr_decode']
                    noise_decode       = sample['noise_decode']
                    snr_custom       = sample['snr_custom']
                    fields = [
                        timestamp, tti, hrq,
                        mcs, prb, tbs,
                        crc, dec_time,
                        cpu, snr, gain, snr_decode, noise_decode, snr_custom
                    ]
                    record = [str(x) for x in fields]
                    file.write('|'.join(record) + '\n')
                    sample_idx += 1
                    if (sample_idx == 10):
                        file.flush()
                        sample_idx = 0
                except queue.Empty:
                    pass
