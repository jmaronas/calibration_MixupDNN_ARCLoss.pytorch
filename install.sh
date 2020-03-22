conda=/usr/local/anaconda3/bin/conda
activate=/usr/local/anaconda3/bin/activate
deactivate=/usr/local/anaconda3/bin/deactivate

$conda create -y --no-default-packages --prefix ~/my_envs/ARD_python3.7_pytorch1.0.0_cuda100 python=3.7
source $activate  ~/my_envs/ARD_python3.7_pytorch1.0.0_cuda100
$conda install -y pytorch=1.0.0 cuda100 torchvision -c pytorch
pip install progress scipy==1.2.1
source $deactivate

cd ~
git clone -b version-1.0.0 https://github.com/jmaronas/pytorch_library.git




