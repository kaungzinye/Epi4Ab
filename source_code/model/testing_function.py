import numpy as np
import pandas as pd
import os
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
from tqdm import tqdm
from source_code.evaluation_and_plot.networkx import plot_network

def prediction_test(data, modelBuild, device):
    data.to(device)
    result = modelBuild(data.x, data.x_seq, data.edge_index, data.edge_attr, data.x_ab, 
                        data.ab_padding_mask, data.feature_token, data.node_size)
    # trueY = data.y == 0
    return result

def record_test(trueY, predY, softY=None, cips_evaluate = False):
    true_interface, pred_interface = trueY.detach().cpu().numpy(), predY.detach().cpu().numpy()
    if cips_evaluate:
        true_interface = np.where(true_interface == 1, True, False)
        pred_interface = np.where(pred_interface == 1, True, False)
        # Handle single-class case (all True or all False in ground truth)
        n_unique_true = len(np.unique(true_interface))
        if n_unique_true < 2:
            # ROC AUC undefined for single class - use placeholder
            roc_auc_score = 0.5  # Random baseline
            average_precision_score = np.mean(true_interface) if np.any(true_interface) else 0.0
        else:
            roc_auc_score = sk_metrics.roc_auc_score(true_interface,pred_interface)
            average_precision_score = sk_metrics.average_precision_score(true_interface,pred_interface)
        recall_score = sk_metrics.recall_score(true_interface,pred_interface, zero_division=0, average='binary')
        precision_score = sk_metrics.precision_score(true_interface,pred_interface, zero_division=0, average='binary')
        f1_score = sk_metrics.f1_score(true_interface,pred_interface, zero_division=0, average='binary')
        accuracy_score = sk_metrics.accuracy_score(true_interface,pred_interface)
    else:
        n_classes = len(np.unique(true_interface))
        if n_classes <= 2:
            # Binary classification: convert to boolean (epitope vs non-epitope)
            true_binary = true_interface != 0
            pred_binary = pred_interface != 0
            if n_classes == 1:
                # All same class - metrics undefined, use placeholders
                roc_auc_score = 0.5  # Random baseline
                average_precision_score = np.mean(true_binary) if np.mean(true_binary) > 0 else 0.0
            else:
                roc_auc_score = sk_metrics.roc_auc_score(true_binary, pred_binary)
                average_precision_score = sk_metrics.average_precision_score(true_binary, pred_binary)
        else:
            # Multi-class (3 classes: 0, 1, 2)
            if softY is not None:
                soft_pred_interface = softY.detach().cpu().numpy()
                # Ensure classes in y_true match columns in y_score
                present_classes = np.unique(true_interface)
                if len(present_classes) < soft_pred_interface.shape[1]:
                    # Filter to only classes present in true labels
                    roc_auc_score = sk_metrics.roc_auc_score(true_interface, soft_pred_interface[:, present_classes], 
                                                             average='macro', multi_class='ovo', labels=present_classes)
                else:
                    roc_auc_score = sk_metrics.roc_auc_score(true_interface, soft_pred_interface, average='macro', multi_class='ovo')
                average_precision_score = sk_metrics.average_precision_score(true_interface, soft_pred_interface, average='micro')
            else:
                roc_auc_score = sk_metrics.roc_auc_score(true_interface, pred_interface, average='macro', multi_class='ovo')
                average_precision_score = sk_metrics.average_precision_score(true_interface, pred_interface, average='micro')
            '''
            For example recall scrore
                Micro: Sum of absolute tp and fn/fp of each classes.
                Macro: Average of recall of all classes.
            '''
        # Use Micro cause average might make the result biased towards outliers in case of imbalanced data. 
        recall_score = sk_metrics.recall_score(true_interface,pred_interface, zero_division=0, average='macro')
        precision_score = sk_metrics.precision_score(true_interface,pred_interface, zero_division=0, average='macro')
        f1_score = sk_metrics.f1_score(true_interface,pred_interface, zero_division=0, average='macro')
        accuracy_score = sk_metrics.accuracy_score(true_interface,pred_interface)

    return [recall_score,
            precision_score,
            f1_score,
            accuracy_score,
            roc_auc_score,
            average_precision_score]
    
def record_res_id(predY, softY, resID, resShort, pdb, fold_folder, trueY=None):
    if trueY is not None:
        df = pd.DataFrame({'res_id':resID,
                        'true_y':trueY,
                        'pred_y':predY,
                        'prob_gt':softY[np.arange(len(softY)),trueY],
                        'prob_0':softY[:,0],
                        'prob_1':softY[:,1],
                        'prob_2':softY[:,2]})
        # df.to_parquet(os.path.join(fold_folder, f'{pdb}.parquet'))
        df.to_csv(os.path.join(fold_folder, f'{pdb}.txt'), sep='\t', index=False)  # Save as txt file
    score = np.log(softY/(1-softY + 1e-6))
    final_df = pd.DataFrame({'res_id':resID,
                            'res_name':resShort,
                            'pred_label':predY,
                            'prob.':softY[np.arange(len(softY)),predY],
                            'score':score[np.arange(len(score)),predY]})
    final_df.to_csv(os.path.join(fold_folder, f'{pdb}_final_result.txt'), sep='\t', index=False)  # Save as txt file

def record_res_id_regression(resID, resShort, pred_score, pdb, fold_folder, trueY=None, output_activation='identity'):
    final_df = pd.DataFrame({
        'res_id': resID,
        'res_name': resShort,
        'pred_score': pred_score
    })
    if output_activation == 'sigmoid':
        final_df['pred_prob'] = pred_score
    if trueY is not None:
        final_df['true_score'] = trueY
    final_df.to_csv(os.path.join(fold_folder, f'{pdb}_final_result.txt'), sep='\t', index=False)

def record_test_pdb(data, predY, softY, res_short, pdb, fold_folder, plot_network_check, networkx_seed):
    if data.y is not None:
        true_y = data.y.detach().cpu().numpy()
    else:
        true_y = None
    pred_y = predY.detach().cpu().numpy()
    soft_y = softY.detach().cpu().numpy()
    res_id = data.res_id.detach().cpu().numpy()
    record_res_id(pred_y, soft_y, res_id, res_short, pdb, fold_folder, true_y)
    if plot_network_check:
        plot_network(res_id, pred_y, true_y, data.edge_index, data.edge_attr, pdb, fold_folder, networkx_seed)

def record_test_pdb_regression(data, pred_score, res_short, pdb, fold_folder, output_activation='identity'):
    if data.y is not None:
        true_y = data.y.detach().cpu().numpy()
    else:
        true_y = None
    res_id = data.res_id.detach().cpu().numpy()
    record_res_id_regression(res_id, res_short, pred_score, pdb, fold_folder, true_y, output_activation=output_activation)

@torch.no_grad()
def test_model(modelBuild, testData, testList, logging, testType:str='train', foldInd=None):
    '''
    testType = ['train', 'validate','test']
    '''
    if logging.train_all in ['yes','with_validation']:
        info_list = [logging.model_name]
        fold_folder = logging.directory_test_record
        iter_desc = f'Testing {testType} pdbs'
    else:
        info_list = [foldInd, logging.model_name]
        fold_folder = os.path.join(logging.directory_test_record, f'fold_{foldInd}')
        if not os.path.exists(fold_folder):
            os.mkdir(fold_folder)
        iter_desc = f'Testing {testType} pdbs, fold {foldInd}'
    record_data = []
    modelBuild.eval()
    print(f"DEBUG: testType is {testType}, logging.train_all is {logging.train_all}")
    tqdm_enum = tqdm(zip(testList, testData), total=len(testList), desc = iter_desc, unit='pdb')
    if logging.loss_function in ['cross_entropy','hce']:
        for pdb, data in tqdm_enum:
            result = prediction_test(data, modelBuild, logging.device)
            pred_y = result.argmax(dim = 1)
            soft_y = F.softmax(result, dim=1)
            
            if not data.y is None:
                assert not data.y.isnan().any(), f'There is NaN value of pdb "{pdb}" in true Y {data.y}'
                record_list = record_test(data.y, pred_y, soft_y)
                assert not pred_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in pred Y {pred_y}'
                assert not soft_y.isnan().any(), f'There is NaN value of pdb "{pdb}" in softmax Y {soft_y}'
                
                record_list = [pdb] + info_list + record_list + [testType] + ['all']
                record_data.append(record_list)
                # CIPS only evaluation
                cips_record_list = record_test(data.y, pred_y, cips_evaluate = True )
                cips_record_list = [pdb] + info_list + cips_record_list + [testType] + ['cips']
                record_data.append(cips_record_list)
            if testType == 'test':
                print(f"Calling record_test_pdb for {pdb} in {fold_folder}")
                record_test_pdb(data, pred_y, soft_y, data.res_short, pdb, fold_folder, logging.plot_network, logging.networkx_seed)
      
    elif logging.loss_function in ['mse','mse_pearson']:
        for pdb, data in tqdm_enum:
            result = prediction_test(data, modelBuild, logging.device)
            pred_score = result.reshape(-1)
            true_y = data.y.reshape(-1).float()
            assert not data.y.isnan().any(), f'There is NaN value of pdb "{pdb}" in true Y {data.y}'
            assert not pred_score.isnan().any(), f'There is NaN value of pdb "{pdb}" in pred Y {pred_score}'
            mse_value = torch.mean((pred_score - true_y) ** 2).item()
            record_list = [mse_value]
            record_list = [pdb] + info_list + record_list + [testType] + ['all']
            record_data.append(record_list)
            if testType == 'test':
                output_activation = getattr(logging, 'output_activation', 'identity')
                record_test_pdb_regression(data, pred_score.detach().cpu().numpy(), data.res_short, pdb, fold_folder, output_activation=output_activation)
                 
    return record_data
