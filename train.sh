python train.py --data_dir data/split/split-1 --name test_0403  --checkpoints_dir /checkpoints --save_epoch_freq 50 --display_freq 10 --save_latest_freq 100 --batchSize 128 --learning_rate_decrease_itr 10 --niter 1000 --lr_visual 0.0001 --lr_audio 0.001 --nThreads 2 --gpu_ids 0 --validation_on --validation_freq 100 --validation_batches 50 --tensorboard True > log/test_0403.log