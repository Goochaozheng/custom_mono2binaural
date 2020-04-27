python train.py --data_dir data/split-8 --name parallel_model_3 --checkpoints_dir checkpoints --save_epoch_freq 50 --display_freq 5 --save_latest_freq 100 --batch_size 128 --learning_rate_decrease_itr 10 --niter 1000 --lr_audio 0.001 --lr_visual 0.0001 --nThreads 0 --validation_on True --validation_freq 100 --validation_batches 50 > log/parallel_model_3.log