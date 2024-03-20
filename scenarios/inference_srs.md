### Steps to benchmark SRS default scheduler.

For the steps to benchmark SRS default scheduler, please refer to [Steps to Collect Data](data_collection.md), with exceptions on steps 3, 4 and 5.

3. **Initialize srsENB**:
     ``` bash
     srsenb ~/.config/srsran/enb.conf \
     --enb_files.sib_config ~/.config/srsran/sib.conf \
     --enb_files.rr_config ~/.config/srsran/rr.conf \
     --enb_files.rb_config ~/.config/srsran/rb.conf \
     --scheduler.policy=time_rr \
     --rf.device_name=zmq \
     --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2101,rx_port=tcp://localhost:2100,id=enb,base_srate=23.04e6"
     ```

4. **Initialize ATHENA Scheduler in SRS Scheduling Mode**:
     ``` bash
     python3 athena_ml.py -m srs -r <results_file> --actions 2 --verbose 1
     ```

5. **Initialize Wireless Channel**:
     ``` bash
     python3 gnuradio/wireless_channel.py --mode=cmd \
     --enb_tx=tcp://localhost:2101 \
     --enb_rx=tcp://*:2100 \
     --ue_tx=tcp://localhost:2001 \
     --ue_rx=tcp://*:2000
     ```
     Confirm that the UE appears connected and that the EPC has confirmed its registration.

     After executing Step 7, enter the new context state in stdin typing: ```beta=<beta>,gain=<gain>``` where ``<beta>`` the congestion factor and ``<gain>`` the gain of the wireless channel.




