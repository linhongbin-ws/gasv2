####################################
## modify to your setting
ANACONDA_PATH="$HOME/miniconda3"
ENV_NAME=gasv2_surrol
######################


source $ANACONDA_PATH/bin/activate
conda activate $ENV_NAME
alias python=$ANACONDA_PATH/envs/$ENV_NAME/bin/python3.9
export LD_LIBRARY_PATH=$ANACONDA_PATH/envs/$ENV_NAME/lib/:/usr/local/lib/