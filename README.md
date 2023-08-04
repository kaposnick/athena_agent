<img src="uc3m.png">
<img src="imdea.png">

<b>ATHENA: Machine Learning and Reasoning for Radio Resources Scheduling in vRAN systems</b> 

The repository contains the code of the paper "ATHENA: Machine Learning and Reasoning for Radio Resources Scheduling in vRAN systems" by N. Apostolakis, M. Gramaglia, L.E. Chatzieleftheriour, T. Subramanya, A. Banchs, H. Sanneck. 

The software has been developed and tested with the following versions:
| Software | Version |
| -------- | ------- |
| OS       | Ubuntu 22.04 |
| Kernel   | 5.15.0-58.generic  |

### 1. Clone the repository
```shell
git clone --recurse-submodules git@github.com:/kaposnick/athena_agent.git
```
### 2. Local installation
Install srsRAN

```shell
cd srsRAN
sudo apt-get update -y && \ 
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
./srsran_install_configs.sh user
```

