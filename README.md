# gasv2


## 2.2. Download and Install

### surrol
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

## Evaluate on dVRK

```sh
python ./run/dvrk_eval.py --reload-dir ./log/2024_10_02-19_18_51@ras-gasv2_surrol-dsa1@dreamerv2-gasv2-high_oracle-train_every1@seed0  --reload-envtag gasv2_dvrk dsa1 --online-eval --visualize --vis-tag obs rgb dsa mask --online-eps 20 --save-prefix xxx
```