# ML4Astrophysics


module load python cuda gcc 
python -m venv pyvenv_ml_cpu --system-site-packages
source pyvenv_ml_cpu/bin/activate
pip install -r requirements_cpu.txt
deactivate

python -m venv pyvenv_ml_gpu --system-site-packages
source pyvenv_ml_gpu/bin/activate
pip install -r requirements_cpu.txt
pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 cuml-cu12 cupy-cuda12x
deactivate

