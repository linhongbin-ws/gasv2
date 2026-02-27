# gasv2

miniconda
```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
sudo chmox +x Miniconda3-xx.sh
./Miniconda3-xx.sh
```


```sh
git clone https://github.com/linhongbin-ws/gasv2.git -b sim
cd gasv2
git submodule update --init --recursive
```


## 2.2. Simulation

### surrol + baselines (except PPO, GASv2-BC)
- Download and install [Miniconda](https://docs.anaconda.com/miniconda/).

- Edit environment variables, go to [init_surrol.sh](./bash/init_surrol.sh) and edit your environment variables.

- Create virtual environment in anaconda
    ```sh
    source bash/init_surrol.sh 
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```

- Install package 
    ```sh
    source bash/init_surrol.sh
    conda install cudnn=8.2 cudatoolkit=11.3 libffi==3.3 ffmpeg -c anaconda -c conda-forge -y
    pushd ext/SurRoL/ && python -m pip install -e . && popd # install surrol
    pushd ext/dreamerv2/ && python -m pip install -e . && popd # install dreamerv2
    python -m pip install -e . # install gym_ras
    ```

### surrol + PPO
- Download and install [Miniconda](https://docs.anaconda.com/miniconda/).

- Edit environment variables, go to [init_surrol_ppo.sh](./bash/init_surrol.sh) and edit your environment variables.

- Create virtual environment in anaconda
    ```sh
    source bash/init_surrol_ppo.sh 
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```

- Install package 
    ```sh
    source bash/init_surrol_ppo.sh
    pushd ext/SurRoL/ && python -m pip install -e . && popd # install surrol
    pushd ext/stable-baselines3/ && python -m pip install -e . && popd # install stable-baselines3
    pushd ext &&  git clone https://github.com/facebookresearch/r3m && cd r3m && python -m pip install -e . && popd # install r3m
    python -m pip install -e . # install gym_ras
    ```

### surrol + GASv2-BC
- install

    ```sh
    source ~/miniconda3/bin/activate
    conda create -n dreamer_fd python=3.7 -y
    conda activate dreamer_fd
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y
    pushd ext/SurRoL/ && python -m pip install -e . && popd # install surrol
    pip install tensorflow==2.9.0 tensorflow_probability==0.17.0 protobuf==3.20.1
    pushd ext/DreamerfD/ && python -m pip install -e . && popd # install DreamerfD
    python -m pip install -e .
    ```


## Real Robot
### dvrk
- Download and install [Miniconda](https://docs.anaconda.com/miniconda/).
- Download [pretrain_models](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155097177_link_cuhk_edu_hk/Elg2xxj3URNJhm6cCNf8GzwBVJXCOfrtLiL83xXECN_7VQ?e=do3lV8&download=1) and unzip to current directory

- Edit environment variables, go to [init_dvrk.sh](./bash/init_dvrk.sh) and edit your environment variables.

- Create virtual environment in anaconda
    ```sh
    source bash/init_dvrk.sh 
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.9 -y
    ```
- Install Conda Dependency
    ```
    source bash/init_dvrk.sh 
    conda install -c conda-forge ros-rospy wstool ros-sensor-msgs ros-geometry-msgs ros-diagnostic-msgs empy rospkg python-orocos-kdl -y 
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch -y
    conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y 
    conda install ffmpeg libffi==3.3 -y
    ```
- dVRK Dependency
    - for melodic:
    ```sh
    sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev  libbluetooth-dev python-wstool python-vcstool python-catkin-tools
    #conda install -c conda-forge ros-rospy wstool ros-sensor-msgs ros-geometry-msgs ros-diagnostic-msgs -y # additional install for melodic since it uses python2.7, need to reinstall all ros dependency
    ```
    - for noetic:
    ```sh
    sudo apt install libxml2-dev libraw1394-dev libncurses5-dev qtcreator swig sox espeak cmake-curses-gui cmake-qt-gui git subversion gfortran libcppunit-dev libqt5xmlpatterns5-dev libbluetooth-dev python3-pyudev python3-wstool python3-vcstool python3-catkin-tools python3-osrf-pycommon
    ```
- Install ROS dependency in Conda environment
    ```sh
    mkdir -p ext/ros_ws/src 
    pushd ext/ros_ws/src 
    git clone https://github.com/ros/geometry -b $ROS_DISTRO-devel 
    git clone https://github.com/ros/geometry2 -b $ROS_DISTRO-devel
    cd ..
    source /opt/ros/$ROS_DISTRO/setup.bash
    catkin config --cmake-args -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9 -DPYTHON_INCLUDE_DIR=$ANACONDA_PATH/envs/$ENV_NAME/include/python3.9 -DPYTHON_LIBRARY=$ANACONDA_PATH/envs/$ENV_NAME/lib/libpython3.9.so
    catkin build # compile ros packages
    popd
    ```
- Install dVRK
    ```sh
    python -m pip install defusedxml numpy==1.23
    mkdir ./ext/dvrk_2_1/src
    pushd ./ext/dvrk_2_1/
    catkin init
    catkin config --cmake-args -DPYTHON_EXECUTABLE=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9 -DPYTHON_INCLUDE_DIR=$ANACONDA_PATH/envs/$ENV_NAME/include/python3.9 -DPYTHON_LIBRARY=$ANACONDA_PATH/envs/$ENV_NAME/lib/libpython3.9.so
    cd src
    vcs import --input https://raw.githubusercontent.com/jhu-saw/vcs/main/ros1-dvrk-2.1.0.vcs --recursive
    catkin build --summary
    popd
    ```
- DualShock
    ```sh
    source ./init_dvrk.sh
    pushd ./ext/
    git clone https://github.com/naoki-mizuno/ds4drv --branch devel
    cd ./ds4drv
    python -m pip install -e .
    sudo cp udev/50-ds4drv.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    popd
    pushd ./ext/ros_ws/src
    git clone https://github.com/naoki-mizuno/ds4_driver.git -b noetic-devel # Do not need to modify for melodic user, use noetic branch to support python3
    catkin build
    popd
    ```

- Install package 
    ```sh
    pip install timm==0.5.4 easydict opt-einsum
    conda install -c conda-forge  python-orocos-kdl -y
    python -m pip install -e . # install gym_ras
    ``````


# Run

## Training

if you need to specifc which GPU, you need to put `CUDA_VISIBLE_DEVICES=0` in fronnt of training command

- GASV2
    ```sh
    python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2  --baseline-tag gas high_oracle3 
    ```
- GASV1
    ```sh
    python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  gasv1 --baseline-tag gas eval_less high_oracle3 
    ```

- dreamerv2
    ```sh
    python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  raw_env --baseline-tag  gas eval_less high_oracle3 
    ```

- GASv2-rawRGB
    ```sh
    python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  no_dsa --baseline-tag gas eval_less high_oracle3 
    ```
- GASv2-rawControl
    ```sh
    python ./run/rl_train.py --env-tag  domain_random_enhance dsa_occup2  no_pid --baseline-tag gas eval_less high_oracle3
    ```

- PPO
    ```sh
    python ./run/rl_train.py --env-tag domain_random_enhance dsa_occup2  raw_env --baseline ppo --baseline-tag high_oracle3
    ```

- GASv2-BC
    ```sh
    python ./run/rl_train.py  --env-tag domain_random_enhance dsa_occup2  --baseline-tag gas --baseline dreamerv2_bc
    ```


## eval on surrol

- Performance Study
    ```sh
    python ./run/rl_train.py --reload-dir ./data/agent/2025_02_13-22_25_34@grasp_any_v2-domain_random_enhance-dsa_occup2@dreamerv2-gas-high_oracle3@seed0  --reload-envtag  domain_random_enhance dsa_occup2  --online-eval --novis --vis-tag obs rgb dsa mask --online-eps 100 --save-prefix xxx --seed 4
    ```


## eval on dvrk
```sh
python ./run/dvrk_eval.py --reload-dir ./data/agent/2024_12_30-10_58_55@grasp_any_v2@dreamerv2-gasv2@seed0  --reload-envtag  gasv2_dvrk --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx
```

```sh
python ./run/dvrk_eval.py --reload-dir ./log/2025_01_02-14_02_12@grasp_any_v2-action_continuous@dreamerv2-gasv2@seed0/  --reload-envtag  gasv2_dvrk action_continuous --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx
```

