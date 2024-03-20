### Steps to Collect Data

Before starting, ensure you have determined the SGi TUN interface IP address to be used by the srsEPC and the target server IP for the UE data collection campaign. Run all binaries with root privileges.

1. **Initialize srsEPC**:
  ```bash
  srsepc ~/.config/srsran/epc.conf --hss.db_file ~/.config/srsran/user_db.csv --spgw.sgi_if_addr=<sgiaddress>
  ```

2. **Initialize srsUE**:
  ``` bash
  ip netns add ue
  srsue ~/.config/srsran/ue.conf --gw.netns=ue \
  --log.filename='/tmp/ue.log' \
  --rf.device_name=zmq \
  --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2001,rx_port=tcp://localhost:2000,id=ue,base_srate=23.04e6"
  ```

3. **Initialize srsENB**:
  ``` bash
  srsenb ~/.config/srsran/enb.conf \
  --enb_files.sib_config ~/.config/srsran/sib.conf \
  --enb_files.rr_config ~/.config/srsran/rr.conf \
  --enb_files.rb_config ~/.config/srsran/rb.conf \
  --scheduler.policy=time_sched_ai \
  --rf.device_name=zmq \
  --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2101,rx_port=tcp://localhost:2100,id=enb,base_srate=23.04e6"
  ```

4. **Initialize ATHENA Scheduler in Random Scheduling Mode**:
  ``` bash
  python3 athena_ml.py -m random -r <results_file> --actions 2 --verbose 1
  ```

5. **Initialize Wireless Channel**:
  ``` bash
  python3 gnuradio/wireless_channel.py --mode=loop \
  --enb_tx=tcp://localhost:2101 \
  --enb_rx=tcp://*:2100 \
  --ue_tx=tcp://localhost:2001 \
  --ue_rx=tcp://*:2000
  ```
  Confirm that the UE appears connected and that the EPC has confirmed its registration.

6. **Initialize iperf server**:
  ``` bash
  iperf -u -s
  ```

7. **Initialize UE iperf Client**:
  ``` bash
  ip netns exec ue ip route add default dev tun_srsue && ip netns exec ue iperf -u -c <sgiaddress> -B 20G -t <seconds>
  ```




