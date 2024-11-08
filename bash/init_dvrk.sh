####################################
## modify to your setting
ANACONDA_PATH="$HOME/miniconda3"
ENV_NAME=gasv2_dvrk
ROS_DISTRO=melodic
######################




source $ANACONDA_PATH/bin/activate
conda activate $ENV_NAME
alias python=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/
source ./ext/ros_ws/devel/setup.bash
source ./ext/dvrk_2_1/devel/setup.bash
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/