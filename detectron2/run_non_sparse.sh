python birdnet/train.py --bev_dataset BASE_BIRDNET
python birdnet/train.py --bev_dataset PLANE_BIRDNET
python birdnet/train.py --bev_dataset MEAN_BIRDNET
python birdnet/train.py --bev_dataset STACK_BIRDNET
 
python birdnet/train.py --bev_dataset BASE_BIRDNET --bev_lr 0.01 --use_weights False
python birdnet/train.py --bev_dataset PLANE_BIRDNET --bev_lr 0.01  --use_weights False
python birdnet/train.py --bev_dataset MEAN_BIRDNET  --bev_lr 0.01  --use_weights False
python birdnet/train.py --bev_dataset STACK_BIRDNET --bev_lr 0.01  --use_weights False
