import torch
from torch import nn
import torch.nn.functional as F

class CustomMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.tensor, target: torch.Tensor) -> torch.tensor:
        return F.mse_loss(input.reshape(-1), target.float())

class MSEPearsonLoss(nn.Module):
    """MSE plus a per-complex correlation penalty.

        loss = mse + pearson_weight * (1 - mean_g Pearson_g(pred, target))

    The Pearson term is computed within each complex (graph) using the batch
    assignment vector, so it optimizes the *shape/ranking* of predictions per
    complex. Plain MSE on per-complex min-max targets is minimized by predicting
    the conditional mean, which collapses predictions toward ~0.4; adding the
    correlation term forces the model to track relative burial across residues.
    A single graph (batch=None, used at validation) is treated as one complex.
    Graphs with fewer than two residues or a flat target are skipped because
    Pearson is undefined there.
    """
    def __init__(self, pearson_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.pearson_weight = pearson_weight
        self.eps = eps

    def _pearson(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_c = pred - pred.mean()
        target_c = target - target.mean()
        denom = (torch.sqrt((pred_c * pred_c).sum() + self.eps)
                 * torch.sqrt((target_c * target_c).sum() + self.eps))
        return (pred_c * target_c).sum() / denom

    def _is_flat(self, vals: torch.Tensor) -> bool:
        return bool((vals - vals.mean()).abs().sum() < self.eps)

    def forward(self, input: torch.Tensor, target: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        pred = input.reshape(-1)
        tgt = target.float().reshape(-1)
        mse = F.mse_loss(pred, tgt)
        if self.pearson_weight == 0:
            return mse
        if batch is None:
            if pred.numel() < 2 or self._is_flat(tgt):
                corr_term = pred.new_zeros(())
            else:
                corr_term = 1.0 - self._pearson(pred, tgt)
        else:
            num_graphs = int(batch.max().item()) + 1
            corrs = []
            for g in range(num_graphs):
                mask = batch == g
                if int(mask.sum()) < 2:
                    continue
                t = tgt[mask]
                if self._is_flat(t):
                    continue
                corrs.append(self._pearson(pred[mask], t))
            if len(corrs) == 0:
                corr_term = pred.new_zeros(())
            else:
                corr_term = 1.0 - torch.stack(corrs).mean()
        return mse + self.pearson_weight * corr_term
    
class HierarchicalCELoss(nn.Module):
    def __init__(self, cross_entropy_weight, device):
        super().__init__()
        self.reachability_matrix = torch.tensor([[1,1,1],
                                                 [0,1,0],
                                                 [0,0,1]]).float().to(device)
        '''
        [[1,1,1,1],
         [0,1,0,0],
         [0,0,1,0],
         [0,1,1,1]]
        '''
        self.cross_entropy_weight = torch.tensor(cross_entropy_weight).to(device)
        self.device = device
        
    def forward(self, logits, targets):
        """
        Hierarchical Cross-Entropy loss
        
        Args:
            logits: Raw model predictions (batch_size, num_classes)
            targets: Ground truth class indices (batch_size)
            reachability_matrix: Matrix encoding hierarchical relationships (num_classes, num_classes)
            weight: Optional class weights
        
        Returns:
            Hierarchical Cross-Entropy loss value
        https://github.com/microsoft/hce-classification
        """
        # Convert logits to probabilities using softmax
        cell_type_probs = torch.softmax(logits, dim=-1)
        
        # Propagate probabilities through the hierarchy using the reachability matrix
        cell_type_probs = torch.matmul(cell_type_probs, self.reachability_matrix.T)
        
        # Apply log transform (with numerical stability term) for NLL loss calculation
        cell_type_probs = torch.log(
            cell_type_probs + torch.tensor(1e-6, device=self.device)
        )
        
        # Calculate negative log-likelihood loss with optional class weights
        hce_loss = F.nll_loss(cell_type_probs, targets, weight=self.cross_entropy_weight).to(self.device)
        return hce_loss
    
def get_loss_function(loss_function, 
                      cross_entropy_weight,
                      device,
                      pearson_loss_weight=1.0):
    if loss_function == 'cross_entropy': 
        if cross_entropy_weight:
            return nn.CrossEntropyLoss(weight=torch.tensor(cross_entropy_weight))
        else:
            return nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        return CustomMSELoss()
    elif loss_function == 'mse_pearson':
        return MSEPearsonLoss(pearson_weight=pearson_loss_weight)
    elif loss_function == 'hce':
        return HierarchicalCELoss(cross_entropy_weight, device)
    
