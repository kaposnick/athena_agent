<img src="imdea.png" width="200" float="left" >
<img src="uc3m.png"  width="600">

<b>ATHENA: Machine Learning and Reasoning for Radio Resources Scheduling in vRAN systems</b> 

The repository contains the code of the paper "ATHENA: Machine Learning and Reasoning for Radio Resources Scheduling in vRAN systems" by N. Apostolakis, M. Gramaglia, L.E. Chatzieleftheriou, T. Subramanya, A. Banchs, H. Sanneck. 

The software has been tested with the following versions:
| Software | Version |
| -------- | ------- |
| OS       | Ubuntu 22.04 |
| Kernel   | 5.15.0-58.generic  |
| Python   | 3.8 |
| Tensorflow | 2.9.1 |
| GNU Radio Companion | 3.9 |
| iperf | 2.0.13|
| cset     | 1.6 |


### 1. Clone the repository
```shell
$ git clone --recurse-submodules git@github.com:/kaposnick/athena_agent.git
```
### 2. srsRAN Local installation

```shell
$ cd srsRAN
$ apt-get update -y && \ 
    apt-get install -y software-properties-common \ 
    build-essential \
    cmake \ 
    libfftw3-dev \
    libmbedtls-dev \
    libboost-program-options-dev \
    libconfig++-dev \
    libsctp-dev \
    libzmq3-dev
$ mkdir build
$ cd build
$ cmake ../
$ make
$ make install
$ ./srsran_install_configs.sh user
```
### 3. Python 3.8 Installation
Make sure that Python 3.8 or above is installed, along with <code>numpy</code>, <code>tensorflow</code>, <code>scipy</code>.

### 4. GNU Radio Companion (GRC) installlation 
Follow the GNU Radio Companion official installation guidlines. The Python libraries are needed so that the channel emulation module is run correctly.

### 5. CPU Isolation
Prevent default Linux scheduler from allocating tasks (threads) on a user-predefined CPU set. In this setup I run it on a 12-CPU server and I pick the last 3 CPUs to isolate.

Open with an editor of your choice file <code>/etc/default/grub</code> and set the following parameter to <code>GRUB_CMDLINE_LINUX_DEFAULT="maybe-ubiquity isolcpus=9-11"</code>. After run
```shell
$ update-grub
$ reboot
```
so that the isolation takes effect.

We will create the CPU set that the srsENB PHY threads are going to run on. 
(run as root)

```shell
$ cset shield --kthread on --cpu=9-11
$ cset set user/worker0 --cpu=9
$ cset set user/worker1 --cpu=10
$ cset set user/worker2 --cpu=11
```

verify that the CPU set namespace is created correctly. We can see the user <code>cpuset</code> namespace that is created.
```shell
$ cset set --list -r
cset: 
         Name       CPUs-X    MEMs-X Tasks Subs Path
 ------------ ---------- - ------- - ----- ---- ----------
         root       0-11 y       0 y   142    2 /
         user       9-11 y       0 n     0    3 /user
       system        0-8 y       0 n   426    0 /system
      worker1         10 n       0 n     0    0 /user/worker1
      worker2         11 n       0 n     0    0 /user/worker2
      worker0          9 n       0 n     0    0 /user/worker0
```
 
### 6. Create the UE namespace
srsUE is going to run on a seperate networking namespace with distinct routing tables.

```shell
$ ip netns add ue
```

### 7. Start srsEPC
```shell
$ srsepc .config/srsran/epc.conf \
  --hss.db_file .config/srsran/user_db.csv
  --spgw.sgi_if_addr=127.0.0.1
```

### 8. Start srsUE
```shell
$ srsue .config/srsran/ue.conf \
  --gw.netns=ue
  --log.filename='/tmp/ue.log' \
  --rf.device_name=zmq \
  --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2001,rx_port=tcp://localhost:2000,id=ue,base_srate=23.04e6"
```

### 9. Start srsENB
```shell
$ srsenb .config/srsran/enb.conf \
    --enb_files.sib_config .config/srsran/sib.conf \
    --enb_files.rr_config .config/srsran/rr.conf \
    --enb_files.rb_config .config/srsran/rb.conf \
    --scheduler.policy=time_sched_ai \
    --expert.pusch_beta_factor=1 \
    --rf.device_name=zmq \
    --rf.device_args="fail_on_disconnect=true,tx_port=tcp://*:2101,rx_port=tcp://localhost:2100,id=enb,base_srate=23.04e6"
```
