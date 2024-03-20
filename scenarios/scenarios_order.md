### Steps to Collect Training Data, Train, and Deploy ATHENA Resource Control Model in srsRAN

1. **Run srsRAN in Random Scheduling Mode**:
    - Configure srsRAN to operate in random scheduling mode, where scheduling decisions (such as MCS, PRB) are made based on different contexts like β and SNR. 
    - To span across the whole context range, configure the `wireless_channel` script with the option `-m loop` and modify the `congestion_levels` and `gain_levels` variables in the top of the Python script according to the dynamic scenario you would like to train data for. 
    - Additionally, run the `srsENB` executable using `--scheduler_policy=time_sched_ai` and the `athena_ml` python script using `--mode=random` to enforce the random scheduling policy.
    
    For the steps to collect data, please refer to [Steps to Collect Data](data_collection.md).

2. **Train the Model**:
    - Train the ATHENA resource control model using the collected training data.

3. **Deploy the Model to the srsRAN Scheduler**:
    - Integrate and deploy the trained ATHENA resource control model into the srsRAN scheduler for real-time scheduling decisions. 
    - To do so, run the `wireless_channel` script with the option `-m cmd` and give the current context as a command line input using the form `'beta=<beta>,gain=<gain>'`. 
    - Run the `srsENB` executable using `--scheduler_policy=time_sched_ai` and the `athena_ml` python script using `--mode=athena` specifying the weights of the actor and critic models.
    
    For the steps to infer ATHENA, please refer to [Steps to infer ATHENA](inference_athena.md).

4. **Benchmark to the Default srsRAN Scheduler**:
    - To benchmark against the default scheduler and retrieve data, run the `srsENB` executable using `--scheduler_policy=time_rr` and the `athena_ml` python script using `--mode=srs`.

    For the steps to infer default srsRAN Scheduler, please refer to [Steps to infer SRS](inference_srs.md).
