python train.py --data_dir data/split-8 --name audio_dense_1 --checkpoints_dir checkpoints --save_epoch_freq 100 --save_latest_freq 50 --batch_size 64 --learning_rate_decrease_itr 10 --niter 500 --lr_audio 0.0001 --lr_visual 0.0001 --nThreads 0 --validation_on True --validation_freq 20 --validation_batches 20 > log/audio_dense_1.log