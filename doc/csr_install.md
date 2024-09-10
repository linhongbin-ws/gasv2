```sh
conda env create -f environment.yml 
cd rti_package/connextdds-py/
python configure.py -n ../rtiddslibs/6.0.1.25 -j8 "x64Linux4gcc7.3.0"
python -m pip install -e .
cd ../..
python -m pip install csrk-0.1-py3-none-any.whl 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch -y
conda install cudnn=8.2 cudatoolkit=11.3 -c anaconda -y 
conda install ffmpeg libffi==3.3 -y
pip install timm==0.5.4 easydict opt-einsum
conda install -c conda-forge python-orocos-kdl -y
```