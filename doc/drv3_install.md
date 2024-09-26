    ```sh
    source bash/init_surrol.sh 
    source $ANACONDA_PATH/bin/activate 
    conda create -n $ENV_NAME python=3.10 -y
    ```

- Install package 
    ```sh
    source bash/init_surrol_drv3.sh
    <!-- conda install cudnn=8.2 cudatoolkit=11.3 libffi==3.3 ffmpeg -c anaconda -c conda-forge -y -->
    pushd ext/SurRoL/ && python -m pip install -e . && popd # install surrol
    <!-- pushd ext/dreamerv2/ && python -m pip install -e . && popd # install dreamerv2 -->
    python -m pip install -e . # install gym_ras
    ```