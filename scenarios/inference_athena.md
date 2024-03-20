### Steps to Deploy the ATHENA resource control model.

For the steps to deploy ATHENA trained model, please refer to [Steps to Collect Data](data_collection.md), with exceptions on steps 4 and 5.

4. **Initialize ATHENA Scheduler in ATHENA Scheduling Mode**:
  ``` bash
  python3 athena_ml.py -m athena -r <results_file> --actions 2 --verbose 1 \
  --actor_weights <path_to_actor_weights> \
  --critic_weights <path_to_critic_weights>
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




