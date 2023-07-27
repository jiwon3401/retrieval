
import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
import torch.nn as nn
#from parallel import DataParallelModel, DataParallelCriterion
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, TripletLoss, NTXent, WeightTriplet
from models.ASE_model import ASE
from data_handling.DataLoader import get_dataloader
import wandb
from tensorboardX import SummaryWriter

import platform
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from pprint import PrettyPrinter
from torch.utils.tensorboard import SummaryWriter
from tools.utils import setup_seed, AverageMeter, a2t, t2a
from tools.loss import BiDirectionalRankingLoss, WeightTriplet, TripletLoss, NTXent, VICReg, InfoNCE, InfoNCE_VICReg
import pickle

from models.ASE_model import ASE
from data_handling.DataLoader import get_dataloader


def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)

    # set up logger
    exp_name = config.exp_name
    
    folder_name = '{}_freeze_{}_lr_{}_' \
                  'seed_{}'.format(exp_name, str(config.training.freeze),
                                             config.training.lr,
                                             config.training.seed)

    log_output_dir = Path('outputs', folder_name, 'logging')
    model_output_dir = Path('outputs', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    model = ASE(config)
    model = model.to(device)

    # set up optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=5,threshold=0.005,threshold_mode='abs',min_lr=0.000001,verbose=True)


    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    
    elif config.training.loss == 'ntxent':
        criterion = NTXent()
    
    elif config.training.loss == 'weight':
        criterion = WeightTriplet(margin=config.training.margin)
        
    elif config.training.loss == 'infonce':
        criterion = InfoNCE()
        
    elif config.training.loss == 'infonce+vicreg':
        criterion = InfoNCE_VICReg(info_weight=1,vic_weight=0.4)
        
    elif config.training.loss == 'bidirect': #'contrastive'??
        criterion = BiDirectionalRankingLoss(margin=config.training.margin)



    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')
    main_logger.info(f'Total parameters: {sum([i.numel() for i in model.parameters()])}')

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']

    # training loop
    recall_sum = []

    for epoch in range(ep, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

            audios, captions, audio_ids, _ = batch_data

            # move data to GPU
            audios = audios.to(device)
            audio_ids = audio_ids.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            loss = criterion(audio_embeds, caption_embeds, audio_ids)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())
        
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)

        elapsed_time = time.time() - start_time

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        r1, r5, r10, mAP10, medr, meanr, val_loss = validate(val_loader, model, device, criterion=criterion)
        r_sum = r1 + r5 + r10
        recall_sum.append(r_sum)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/r@1', r1, epoch)
        writer.add_scalar('val/r@5', r5, epoch)
        writer.add_scalar('val/r@10', r10, epoch)
        #writer.add_scalar('val/r@50', r50, epoch)
        writer.add_scalar('val/mAP10', mAP10, epoch)
        writer.add_scalar('val/med@r', medr, epoch)
        writer.add_scalar('val/mean@r', meanr, epoch) 

        # save model
        if r_sum >= max(recall_sum):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict(),
                'optimizer': model.state_dict(),
                'epoch': epoch,
                'config': config
            }, str(model_output_dir) + '/best_model.pth')

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {current_lr:.6f}.')

    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    validate(test_loader, model, device)
    main_logger.info('Evaluation done.')
    writer.close()

@torch.no_grad()
def validate(data_loader, model, device, criterion=None):

    val_logger = logger.bind(indent=1)
    model.eval()
    val_loss = AverageMeter()
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
            # move data to GPU
            audios = audios.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            # Code for validation loss
            if criterion!=None:
                loss = criterion(audio_embeds, caption_embeds, audio_ids)
                val_loss.update(loss.cpu().item())

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()

        val_logger.info(f'Validation loss: {val_loss.avg :.3f}')
        # evaluate text to audio retrieval
        r1, r5, r10, mAP10, medr, meanr = t2a(audio_embs, cap_embs)

        val_logger.info('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1, r5, r10, mAP10, medr, meanr))
        

        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a = a2t(audio_embs, cap_embs)

        val_logger.info('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a))

        return r1, r5, r10, mAP10, medr, meanr, val_loss.avg

    
 




'''
def train(config):

    # setup seed for reproducibility
    setup_seed(config.training.seed)
    # set up logger
    exp_name = config.exp_name
    folder_name = '{}_data_{}_freeze_{}_lr_{}_' \
                  'margin_{}_seed_{}'.format(exp_name, config.dataset,
                                             str(config.training.freeze),
                                             config.training.lr,
                                             config.training.margin,
                                             config.training.seed)
    ###config.training.loss, config.cnn_encoder.model
    

    log_output_dir = Path('outputs', folder_name, 'logging')
    model_output_dir = Path('outputs', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    main_logger = logger.bind(indent=1)

    # setup TensorBoard
    writer = SummaryWriter(log_dir=str(log_output_dir) + '/tensorboard')
    

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    # set up model
    device, device_name = ('cuda',
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ('cpu', platform.processor())
    main_logger.info(f'Process on {device_name}')

    model = ASE(config)
    model = model.to(device)
#     model = ASE(config).cuda()
#     model = nn.DataParallel(model).cuda()
    
    ###########

    # set up optimizer and loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    if config.training.loss == 'triplet':
        criterion = TripletLoss(margin=config.training.margin)
    elif config.training.loss == 'ntxent':
        criterion = NTXent()
    elif config.training.loss == 'weight':
        criterion = WeightTriplet(margin=config.training.margin)
    else:
        criterion = BiDirectionalRankingLoss(margin=config.training.margin)

    # set up data loaders
    train_loader = get_dataloader('train', config)
    val_loader = get_dataloader('val', config)
    test_loader = get_dataloader('test', config)

    main_logger.info(f'Size of training set: {len(train_loader.dataset)}, size of batches: {len(train_loader)}')
    main_logger.info(f'Size of validation set: {len(val_loader.dataset)}, size of batches: {len(val_loader)}')
    main_logger.info(f'Size of test set: {len(test_loader.dataset)}, size of batches: {len(test_loader)}')

    ep = 1

    # resume from a checkpoint
    if config.training.resume:
        checkpoint = torch.load(config.path.resume_model)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['epoch']

    # training loop
    recall_sum = []

    for epoch in range(ep, config.training.epochs + 1):
        main_logger.info(f'Training for epoch [{epoch}]')

        epoch_loss = AverageMeter()
        start_time = time.time()
        model.train()

        for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):

            audios, captions, audio_ids, _ = batch_data

            # move data to GPU
            audios = audios.to(device)
            audio_ids = audio_ids.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            loss = criterion(audio_embeds, caption_embeds, audio_ids)
            # audio_ids = labels

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad)
            optimizer.step()

            epoch_loss.update(loss.cpu().item())
            
            #wandb.log(epoch_loss) #  ValueError:wandb.log must be passed a dictionary
            
        writer.add_scalar('train/loss', epoch_loss.avg, epoch)

        elapsed_time = time.time() - start_time

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {epoch_loss.avg:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {scheduler.get_last_lr()[0]:.6f}.')
        

        # validation loop, validation after each epoch
        main_logger.info("Validating...")
        r1, r5, r10, mAP10, medr, meanr = validate(val_loader, model, device) ###error
        r_sum = r1 + r5 + r10
        recall_sum.append(r_sum)

        writer.add_scalar('val/r@1', r1, epoch)
        writer.add_scalar('val/r@5', r5, epoch)
        writer.add_scalar('val/r@10', r10, epoch)
        #writer.add_scalar('val/r@50', r50, epoch)
        writer.add_scalar('val/mAP10', mAP10, epoch)
        writer.add_scalar('val/med@r', medr, epoch)
        writer.add_scalar('val/mean@r', meanr, epoch)

        # save model  ## 더추가해도될듯???? loss추가
                      ## mAP10 기준으로 해야할듯. 
        if r_sum >= max(recall_sum):
            main_logger.info('Model saved.')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': epoch_loss.avg,
            }, str(model_output_dir) + '/best_model.pth')

            
            
            #f"{str(model_output_dir)} + '/bestmodel_epoch_{epoch}_loss_{epoch_loss.avg}.pth")
            ###### 수정하기!! ######
          
            #f"{str(model_output_dir) + '/checkpoint_epoch_{epoch}_loss_{epoch_loss.avg}.pth'}
            
            
            'torch.save({'epoch':e,
                            'model_state_dict':model.state_dict(), 
                            'optimizer_state_dict':optimizer.state_dict(),
                            'loss':epoch_loss,}, f"saved/checkpoint_model_{e}_{epoch_loss/len(dataloader)}_{epoch_acc/len(dataloader)}.pt")
            
            
        scheduler.step()
        

        
    # Training done, evaluate on evaluation set
    main_logger.info('Training done. Start evaluating.')
    
    # best_checkpoint = torch.load(f"{str(model_output_dir)} + '/bestmodel_epoch_{epoch}_loss_{epoch_loss.avg}.pth")
    
    best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth')
    
    model.load_state_dict(best_checkpoint['model'])
    best_epoch = best_checkpoint['epoch']
    main_logger.info(f'Best checkpoint occurred in {best_epoch} th epoch.')
    validate(test_loader, model, device)
    #####위에서 validation 계산
    
    main_logger.info('Evaluation done.')
    writer.close()


def validate(data_loader, model, device):

    val_logger = logger.bind(indent=1)
    model.eval()
    
    #eval_loss, eval_steps = 0.0, 0
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None

        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs = batch_data
 
            # move data to GPU
            audios = audios.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()
            
            # loss = criterion(audio_embeds, caption_embeds, audio_ids)
            # eval_loss += loss.cpu().numpy()
            # eval_steps +=1

            
        # val_logger.info('validation loss: {:.3f}'.format(eval_loss/(eval_steps + le-20)))
        # return r1, r5, r10, mAP10, medr, meanr, eval_loss/(eval_steps + le-20)
        
        # evaluate text to audio retrieval
        r1, r5, r10, mAP10, medr, meanr = t2a(audio_embs, cap_embs)

        val_logger.info('Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10: {:.2f}, medr: {:.2f}, meanr: {:.2f}'.format(
                         r1, r5, r10, mAP10, medr, meanr))


        # evaluate audio to text retrieval
        r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a = a2t(audio_embs, cap_embs)

        val_logger.info('Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                        'r10: {:.2f}, mAP10_a: {:.2f},  medr: {:.2f}, meanr: {:.2f}'.format(
                         r1_a, r5_a, r10_a, mAP10_a, medr_a, meanr_a))
        

        return r1, r5, r10, mAP10, medr, meanr
'''
