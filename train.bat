python train.py --data_dir data/split-8 --name split_2 --trained_weights checkpoints\split_1\model_latest.pth --checkpoints_dir checkpoints --save_epoch_freq 50 --display_freq 20 --save_latest_freq 20 --batch_size 200 --learning_rate_decrease_itr 50 --niter 1000 --lr_audio 0.002 --lr_visual 0.0002 --nThreads 0 --validation_on True --validation_freq 20 --validation_batches 50 > log/split_2.log