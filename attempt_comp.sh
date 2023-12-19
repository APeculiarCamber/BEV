
# Excepts python version 3.10????

pip install torch
pip install opencv-python
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install torchvision
pip install fvcore

git clone https://github.com/AlejandroBarrera/birdnet2/

cd birdnet2

sed -i 's/AT_CHECK(/TORCH_CHECK(/g' detectron2/layers/csrc/deformable/deform_conv.h
sed -i 's/AT_CHECK(/TORCH_CHECK(/g' detectron2/layers/csrc/deformable/deform_conv_cuda.cu 
sed -i 's/Image.LINEAR/Image.BILINEAR/g' detectron2/data/transforms/transform.py

sed -i 's/from collections import Mapping, OrderedDict/\
from collections import OrderedDict\
from collections.abc import Mapping\
/g' detectron2/evaluation/testing.py

sed -i 's/# Paths/\
os.putenv("DETECTRON_ROOT", os.getcwd())\
os.putenv("PYTHON_PATH", os.getcwd())\
/g' tools/train_net_BirdNetPlus.py


echo *********************** DID SEDs *****************************

python -m pip install -e .
