import torch

def set_optimizer(modelBuild, logging, learning_rate=None):
    '''
    This function for setting optimizer.
    '''
    optimizer_method = logging.optimizer_method.lower()
    assert optimizer_method in ['adam',
                                'momentum',
                                'sgd'], f'{logging.optimizer_method} is not defined.'
    lr = logging.learning_rate if learning_rate is None else learning_rate
    if optimizer_method == 'adam':
        return torch.optim.Adam(modelBuild.parameters(), 
                                lr = lr, 
                                weight_decay = logging.weight_decay)
    elif optimizer_method == 'momentum':
        return torch.optim.SGD(modelBuild.parameters(), 
                               lr = lr, 
                               weight_decay = logging.weight_decay, 
                               momentum = logging.momentum)
    elif optimizer_method == 'sgd':
        return torch.optim.SGD(modelBuild.parameters(), 
                               lr = lr, 
                               weight_decay = logging.weight_decay) 
