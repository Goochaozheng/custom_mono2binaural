python train.py --data_dir data/split-8 --name parallel_model_6_2 --checkpoints_dir checkpoints --model_weights checkpoints/parallel_model_6/model_latest.pth --save_epoch_freq 50 --display_freq 10 --save_latest_freq 100 --batch_size 64 --learning_rate_decrease_itr 10 --niter 1000 --lr_audio 0.001 --lr_visual 0.0001 --nThreads 0 --validation_on True --validation_freq 20 --validation_batches 20 > log/parallel_model_6_2.log