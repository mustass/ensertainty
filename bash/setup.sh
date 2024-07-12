echo
python3 -m pip install --upgrade pip
python3 -m venv venv/
source ./venv/bin/activate
python3 -m pip install -e .
python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install urllib3==1.26.6
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false