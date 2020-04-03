# Training process for mono2nbinaural

from dataloader.custom_dataset import CustomDataset
from models.audioVisual_model import AudioVisualModel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
import torchvision
import torch
import os
import time

def create_optimizer(model, opt):
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

# Used to display validation loss
def display_val(model, loss_criterion, writer, index, dataset_val, opt):
    losses = []
    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                loss = loss_criterion(output, val_data['audio_gt'][:,:,:-1,:].cuda())
                losses.append(loss.item()) 
            else:
                break
    avg_loss = sum(losses)/len(losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_loss', avg_loss, index)
    print('val loss: %.3f' % avg_loss)
    return avg_loss 



def main():

    opt = TrainOptions().parse()
    device = torch.device("cuda:0")

    # Construct dataset and dataloader
    dataset = CustomDataset(opt)
    dataloader = DataLoader(
        dataset, 
        batch_size=opt.batchSize, 
        shuffle=True, 
        num_workers=int(opt.nThreads)
    )

    #create validation set data loader if validation_on option is set
    if opt.validation_on:
        #temperally set to val to load val data
        opt.mode = 'val'
        dataset_val = CustomDataset(opt)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=opt.batchSize, 
            shuffle=True, 
            num_workers=int(opt.nThreads)
        )
        dataset_size_val = len(dataloader_val)
        print('#validation clips = %d' % dataset_size_val)
        opt.mode = 'train' #set it back

    # Tensorboard
    if opt.tensorboard:
        writer = SummaryWriter(comment=opt.name)
    
    # Build network
    model = AudioVisualModel(opt)
    model.to(device)

    if opt.tensorboard:
        writer.add_graph(model, next(iter(dataloader)))

    # Create optimizer
    optimizer = create_optimizer(model, opt)

    # Set up loss function
    loss_criterion = torch.nn.MSELoss()
    loss_criterion.cuda(device)

    # Initialization
    total_steps = 0
    data_loading_time = []
    model_forward_time = []
    model_backward_time = []
    batch_loss = []
    best_err = float("inf")
    metric_dict = {
        'best_err': 0,
        'final_loss': 0
    }

    for epoch in range(1, opt.niter+1):
        torch.cuda.synchronize()
        epoch_start_time = time.time()

        if(opt.measure_time):
            iter_start_time = time.time()

        for i, data in enumerate(dataloader):
            if(opt.measure_time):
                torch.cuda.synchronize()
                iter_data_loaded_time = time.time()
            
            total_steps += opt.batchSize

            # Forward
            model.zero_grad()
            output = model.forward(data)

            # Compute loss
            loss = loss_criterion(output, data['audio_diff'][:,:,:-1,:].cuda())
            batch_loss.append(loss.item())  

            if(opt.measure_time):
                torch.cuda.synchronize()
                iter_data_forwarded_time = time.time()

            # Updata optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if(opt.measure_time):
                iter_model_backwarded_time = time.time()
                data_loading_time.append(iter_data_loaded_time - iter_start_time)
                model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)                             

            # Display
            if(total_steps // opt.batchSize % opt.display_freq == 0):
                print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                avg_loss = sum(batch_loss) / len(batch_loss)
                metric_dict['final_loss'] = avg_loss
                print('Average loss: %.3f' % (avg_loss))
                batch_loss = []
                if opt.tensorboard:
                    writer.add_scalar('data/loss', avg_loss, total_steps)
                if(opt.measure_time):
                    print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                    print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                    print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                    data_loading_time = []
                    model_forward_time = []
                    model_backward_time = []
                print('end of display \n')                

            # Save latest
            if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                torch.save(model.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'model_latest.pth'))

            # Validation
            if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
                model.eval()
                opt.mode = 'val'
                print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                val_err = display_val(model, loss_criterion, writer, total_steps, dataloader_val, opt)
                print('end of display \n')
                model.train()
                opt.mode = 'train'
                #save the model that achieves the smallest validation error
                if val_err < best_err:
                    best_err = val_err
                    metric_dict['best_err'] = best_err
                    print('saving the best model (epoch %d, total_steps %d) with validation error %.3f\n' % (epoch, total_steps, val_err))
                    torch.save(model.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'model_best.pth'))

            if(opt.measure_time):
                 iter_start_time = time.time()

        # Save at certain epoch
        if(epoch % opt.save_epoch_freq == 0):
            print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
            torch.save(model.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '_model.pth'))            


        #decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
        if(opt.learning_rate_decrease_itr > 0 and epoch%opt.learning_rate_decrease_itr == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.decay_factor
            print('decreased learning rate by ', opt.decay_factor)        


    if opt.tensorboard:
        writer.add_hparams(vars(opt), metric_dict)

if __name__ == '__main__':
    main()
