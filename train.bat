python train.py --data_dir data/split/split-1 --name batch_200 --checkpoints_dir checkpoints --save_epoch_freq 100 --display_freq 10 --save_latest_freq 100 --batch_size 200 --learning_rate_decrease_itr 10 --niter 1000 --lr_visual 0.0001 --lr_audio 0.001 --nThreads 0 --validation_on True --validation_freq 100 --validation_batches 50 --tensorboard True > log/batch_200.log