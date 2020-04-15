# A wrapper script that trains the SELDnet. The training stops when the SELD error (check paper) stops improving.
# This is only for the training of SED, and DOA with regression strategy.

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
from metrics import evaluation_metrics
import parameter
from tqdm import tqdm
import argparse

import torch
import torch.optim as optim
from pytorch_model import CUDA, kaiming_init, mse_loss, bce_loss, weighted_mse_loss
from pytorch_model import CRNN_SED, MTFA_SED, MCRNN_SED
from torch.autograd import Variable

plot.switch_backend('agg')


def collect_test_labels_3000(_data_gen_test):
    # Collecting ground truth for test data
    nb_batch = _data_gen_test.get_total_batches_in_data()
    batch_size = 1
    gt_sed = np.zeros((nb_batch * batch_size, 3000, 11))

    print('nb_batch in test: {}'.format(nb_batch))
    cnt = 0
    for _, tmp_label in _data_gen_test.generate():
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int)


def save_model(model, model_name='model'):
    states = {'model_states': model.state_dict()}
    with open(model_name, 'wb+') as f:
        torch.save(states, f)


def load_model(model, model_name='model'):
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as f:
            checkpoint = torch.load(f)
        model.load_state_dict(checkpoint['model_states'])
    else:
        raise ValueError('The specified model file does not exists!')
    return model
    

def main(args):
    '''
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the system configuration in parameters.py. (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this. (default) 1
    '''
    # use parameter set defined by user
    dataset, mode, task_id, job_id = args.dataset, args.mode, args.name, args.job_id
    task = 'sed'; feat_type = 'mel'; nb_ch = 4; doa_type = None
    params, model_params = parameter.get_params(dataset=dataset, mode=mode, task_id=task_id, feat_type=feat_type, doa=doa_type)

   
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'eval':
        test_splits = [0, 0, 0, 0]
        val_splits = [91, 91, 91, 91]
        train_splits = [[1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                        [1, 2, 3, 4]]
        

    model_name_list = []

    avg_scores_val = []
    avg_scores_test = []
    for split_cnt, split in enumerate(test_splits):
        trial = split_cnt
        print('\nThis is trial {}'.format(trial+1))

        # Unique name for the run
        model_dir_prefix = os.path.join(params['model_dir'], task) if task == 'sed' else os.path.join(params['model_dir'], 'doa_reg')
        cls_feature_class.create_folder(model_dir_prefix)
        unique_name = '{}{}_{}_{}_sed_eval_split{}'.format(task_id, str(job_id), params['dataset'], params['feat_type'], split_cnt+1)
        model_name_list.append(unique_name)
        unique_name = os.path.join(model_dir_prefix, unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print('\tmodel unique name: {}\n'.format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            dataset=params['dataset'], 
            split=train_splits[split_cnt], 
            batch_size=params['batch_size'],
            seq_len=params['seq_length'], 
            feat_label_dir=params['feat_label_dir'],
            feat_type=feat_type,
            doa=doa_type,
            trial=1
        )
        print('the file length is {}'.format(len(data_gen_train._filenames_list)))
        #assert len(data_gen_train._filenames_list) == 2100

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            dataset=params['dataset'], 
            split=val_splits[split_cnt], 
            batch_size=params['batch_size'],
            seq_len=3000,
            per_file=True,
            feat_label_dir=params['feat_label_dir'],
            shuffle=False,
            feat_type=feat_type,
            doa=doa_type,
            trial=1
        )

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        gt = collect_test_labels_3000(data_gen_val)
        sed_gt = evaluation_metrics.reshape_3Dto2D(gt)     # [3000*100, 11]
        nb_classes = data_gen_train.get_nb_classes()
        def_elevation = data_gen_train.get_default_elevation()
        # Pytorch model
        if task_id == 'crnn':
            model = CUDA(CRNN_SED(data_in, data_out[0]))
        elif task_id == 'mcrnn':
            model = CUDA(MCRNN_SED(data_in, data_out[0]))
        model.apply(kaiming_init)
        
        total_num = sum(param.numel() for param in model.parameters())
        print('==========================================')
        print('Total parameter number for {}: {}'.format(model_params['method'], total_num))
        print('==========================================')
        
        # Pytorch optimizer
        optimizer = optim.Adam(params=model.parameters(), lr=0.001)

        feat_torch = CUDA(Variable(torch.FloatTensor(params['batch_size'], nb_ch, params['seq_length'], params['feat_dim'])))
        label_sed = CUDA(Variable(torch.FloatTensor(params['batch_size'], params['seq_length'], 11)))
        
        best_seld_metric = 99999
        best_sed_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        seld_metric = np.zeros(params['nb_epochs'])
        tr_loss = np.zeros(params['nb_epochs'])
        sed_val_loss = np.zeros(params['nb_epochs'])
        sed_metric = np.zeros((params['nb_epochs'], 2))
        nb_epoch = params['nb_epochs']
        
        # start training
        pbar_epoch = tqdm(total=nb_epoch, desc='[Epoch]')
        for epoch_cnt in range(nb_epoch):
            # train stage
            model.train()
            iter_cnt = 0
            for feat, label in data_gen_train.generate():
                feat_torch.resize_(params['batch_size'], nb_ch, params['seq_length'], params['feat_dim'])
                feat_torch.data.copy_(torch.from_numpy(feat))

                label_sed.resize_(params['batch_size'], params['seq_length'], 11)
                label_sed.data.copy_(torch.from_numpy(label[0]))
                sed = model(feat_torch)

                sed_loss = bce_loss(sed, label_sed)
                doa_loss = 0.0
                
                total_loss = sed_loss + doa_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                if iter_cnt % params['print_iter'] == 0:
                    pbar_epoch.write('Iteration: {:3d}, sed_loss: {:.4f}'.format(iter_cnt, sed_loss))

                #pbar_iteration.update(1)
                iter_cnt += 1
                if iter_cnt >= data_gen_train.get_total_batches_in_data():
                    break
            iter_cnt = 0
            sed_validation_loss = 0
            entire_pred_sed = np.zeros((data_gen_val._batch_size*data_gen_val.get_total_batches_in_data(), 3000, 11))
            
            model.eval()
            with torch.no_grad():
                for feat, label in data_gen_val.generate():
                    batch_size = feat.shape[0]
                    
                    feat_torch.resize_(batch_size, nb_ch, 3000, params['feat_dim'])
                    feat_torch.data.copy_(torch.from_numpy(feat))
                    label_sed.resize_(batch_size, 3000, 11)
                    label_sed.copy_(torch.from_numpy(label[0]))

                    sed = model(feat_torch)
                    sed_loss = bce_loss(sed, label_sed)
                    sed_validation_loss += sed_loss

                    # concat all predictions
                    entire_pred_sed[iter_cnt*batch_size:(iter_cnt+1)*batch_size, :] = sed.detach().cpu().numpy()
                    #pbar_validation.update(1)
                    iter_cnt += 1
                    if iter_cnt >= data_gen_val.get_total_batches_in_data():
                        break

            sed_validation_loss = sed_validation_loss/data_gen_val.get_total_batches_in_data()

            tr_loss[epoch_cnt] = total_loss
            sed_val_loss[epoch_cnt] = sed_validation_loss

            # Calculate the metrics
            sed_pred = evaluation_metrics.reshape_3Dto2D(entire_pred_sed) > params['threshold']  # compared with threshold            
            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_val.nb_frames_1s())
            patience_cnt += 1
            if sed_metric[epoch_cnt, 0] < best_sed_metric:
                best_sed_metric = sed_metric[epoch_cnt, 0]
                best_epoch = epoch_cnt
                save_model(model, model_name)
                patience_cnt = 0

            pbar_epoch.update(1)
            pbar_epoch.write(
                'epoch_cnt: %d, sed_tr_loss: %.4f, sed_val_loss: %.4f, ER_overall: %.2f, F1_overall: %.2f, best_sed_ER: %.4f, best_epoch : %d\n' %
                (
                    epoch_cnt, tr_loss[epoch_cnt], sed_val_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1],
                    best_sed_metric, best_epoch
                )
            )

            if patience_cnt >= params['patience']:
                break

        pbar_epoch.close()

        avg_scores_val.append([sed_metric[best_epoch, 0], sed_metric[best_epoch, 1],  best_seld_metric])
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(sed_metric[best_epoch, 0], sed_metric[best_epoch, 1]))
        
        
        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('Loading testing dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            dataset=params['dataset'], split=split, batch_size=params['batch_size'], seq_len=3000,
            feat_label_dir=params['feat_label_dir'], shuffle=False, per_file=True,
            is_eval=True,  #if params['mode'] is 'eval' else False, 
            feat_type=feat_type, 
            doa=doa_type
        )
        test_batch_size = data_gen_test._batch_size

        print('\nLoading the best model and predicting results on the testing split')
        model = load_model(model, '{}_model.h5'.format(unique_name))
        model.eval()
        
        # test stage
        total_test_batches = data_gen_test.get_total_batches_in_data()
        pbar_test = tqdm(total=total_test_batches, desc='[Testing]')
        iter_cnt = 0
        entire_test_sed = np.zeros((100, 3000, 11))

        with torch.no_grad():
            for feat, label in data_gen_test.generate():
                batch_size = feat.shape[0]

                feat_torch.data.resize_(batch_size, nb_ch, 3000, params['feat_dim'])
                feat_torch.data.copy_(torch.from_numpy(feat))
                    
                sed = model(feat_torch)
                # concat all predictions
                entire_test_sed[iter_cnt*test_batch_size:(iter_cnt+1)*test_batch_size, :] = sed.detach().cpu().numpy()
                pbar_test.update(1)
                iter_cnt += 1
                if iter_cnt >= data_gen_test.get_total_batches_in_data():
                    break

        pbar_test.close()

        test_sed_pred = evaluation_metrics.reshape_3Dto2D(entire_test_sed) > params['threshold']
        _, test_data_out = data_gen_test.get_data_sizes()
        test_gt = collect_test_labels_3000(data_gen_test)
        test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt)
        assert test_sed_pred.shape[0] == test_sed_gt.shape[0]
        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_sed_gt, data_gen_test.nb_frames_1s())
        avg_scores_test.append([test_sed_loss[0], test_sed_loss[1]])
        print('Results on test split:')
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(test_sed_loss[0], test_sed_loss[1]))
        
    
    print('\n\nValidation split scores per fold:\n')
    for cnt in range(len(val_splits)):
        print('\t Trial {} - SED ER: {} F1: {}, model name: {}'.format(
            cnt+1, avg_scores_val[cnt][0], avg_scores_val[cnt][1], model_name_list[cnt]))
    
    print('\n\nTesting split scores per fold:\n')
    for cnt in range(len(test_splits)):
        print('\t Split {} - SED ER: {} F1: {}'.format(
            test_splits[cnt], avg_scores_test[cnt][0], avg_scores_test[cnt][1]))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SELD')
    parser.add_argument('-gpu', default=6, type=int, help='choose gpu number')
    parser.add_argument('-m', '--mode', default='eval', type=str, choices=['dev', 'eval'], help='choose mode')
    parser.add_argument('-d', '--dataset', default='foa', type=str, choices=['foa', 'mic'], help='choose dataset')
    parser.add_argument('-n', '--name', default='crnn', type=str, help='unique name for each method')
    parser.add_argument('-id', '--job_id', default='1', type=str, help='unique output name for a specific method (different parameters)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    try:
        sys.exit(main(args))
    except (ValueError, IOError) as e:
        sys.exit(e)
