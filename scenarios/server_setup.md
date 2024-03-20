### Steps to setup the local machine

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

1. **srsRAN Local installation**:
    ```shell
    cd srsRAN
    apt-get update -y && \ 
     apt-get install -y software-properties-common \ 
     build-essential \
     cmake \ 
     libfftw3-dev \
     libmbedtls-dev \
     libboost-program-options-dev \
     libconfig++-dev \
     libsctp-dev \
     libzmq3-dev
    mkdir build
    cd build
    cmake ../
    make
    make install
    ./srsran_install_configs.sh user```

2. **Python 3.8 Installation**:
Ensure Python 3.8 or a newer version is installed, and make sure to have the following packages installed: <code>numpy</code>, <code>tensorflow</code>, and <code>scipy</code>.

3. **GNU Radio Companion (GRC) installlation**:
Please adhere to the official installation guidelines provided by GNU Radio Companion. Additionally, ensure that the required Python libraries are installed to ensure proper functionality of the channel emulation module.

4. **CPU Isolation**:
To prevent the default Linux scheduler from allocating tasks (threads) on a user-predefined CPU set, follow these steps:

    - Open the file `/etc/default/grub` with your preferred text editor.
    - Set the following parameter:
        ```bash
        GRUB_CMDLINE_LINUX_DEFAULT="maybe-ubiquity isolcpus=9-11"
        ```
    - After editing the file, run the following commands:
        ```bash
        update-grub
        reboot
        ```
        This ensures that the isolation of CPUs takes effect.
    - Next, create the CPU set for the srsENB PHY threads to run on. Run the following commands as root:
        ```shell
        cset shield --kthread on --cpu=9-11
        cset set user/worker0 --cpu=9
        cset set user/worker1 --cpu=10
        cset set user/worker2 --cpu=11
        ```
    - To verify that the CPU set namespace is created correctly, use the following command:
        ```shell
        cset set --list -r
        ```

    - You should see the cpuset namespace for the user as follows:
        ```shell
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
