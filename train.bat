python train.py --data_dir data/split-4 --name mask_modified_split4 --checkpoints_dir checkpoints --save_epoch_freq 50 --display_freq 5 --save_latest_freq 50 --batch_size 64 --learning_rate_decrease_itr 10 --niter 400 --lr_audio 0.001 --nThreads 0 --validation_on True --validation_freq 100 --validation_batches 50 > log/mask_modified_split4.log