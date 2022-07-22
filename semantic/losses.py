import torch
from torch.nn import functional as F
import torch.nn as nn
from scipy.ndimage.morphology import distance_transform_edt

class TestLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ignore_index = -100
        self.reduction = 'mean'
        self.weight = None

    def forward(self, input, target):
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        print(input.shape)
        print(target.shape)
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    # Algorithm 1, calculate the gradient of the Jaccard loss extension
    def gradient(self, sorted_labels):
        sum = sorted_labels.sum()
        intersection = sum - sorted_labels.float().cumsum(0)
        union = sum + (1 - sorted_labels).float().cumsum(0)
        g = 1 - intersection / union
        p = sorted_labels.shape[0]
        if p > 1:
            g[1:p] = g[1:p] - g[0:-1]
        return g

    def forward(self, input, target):
        device = input.device
        if device == 'cpu':
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        classes = input.shape[1]
        # Permute to be b, x, y, z, c
        input_permuted = input.permute(0, 2, 3, 4, 1).contiguous()
        # Flatten from view of classes
        input_flattened = input_permuted.view(-1, classes)
        target_flattened = target.view(-1)
        losses = []
        for i in range(classes):
            class_target = (target_flattened == i).float()
            class_input = input_flattened[:, i]
            class_loss = (class_target.requires_grad_() - class_input).abs()
            class_loss_sorted, class_loss_index = torch.sort(class_loss, 0, descending=True)
            sorted_labels = class_target[class_loss_index]
            losses.append(class_loss_sorted @ self.gradient(sorted_labels).requires_grad_())
        # Can't use torch.Tensor otherwise you lose the gradient properties
        losses = torch.stack(losses)
        if self.reduction == 'mean':
            return losses.mean()
        else:
            return losses.sum()

class ActiveBoundaryLoss(nn.Module):

    def __init__(self, theta=20):
        super().__init__()
        self.ignore_index = -100
        self.reduction = 'mean'
        self.weight = None
        self.directions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    self.directions.append([i, j, k])
        self.directions.remove([0, 0, 0])
        self.theta = theta

    def forward(self, input, target):
        # ABL Phase I
        # Detect PDBs
        device = input.device
        if device == torch.device('cpu'):
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        kldiv = nn.KLDivLoss(reduction='none')
        cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        # Channels last
        input_preds = input.permute((0, 2, 3, 4, 1))

        # Compute our KL Divergence
        kl_vals = torch.zeros(input_preds.shape[:-1])

        horiz = torch.zeros(input_preds.shape)
        horiz[:, :, :, :-1, :] = input_preds[:, :, :, 1:, :]

        vert = torch.zeros(input_preds.shape)
        vert[:, :, :-1, :, :] = input_preds[:, :, 1:, :, :]

        depth = torch.zeros(input_preds.shape)
        depth[:, :-1, :, :, :] = input_preds[:, 1:, :, :, :]

        kldiv_horiz = kldiv(input_preds, horiz)
        kldiv_horiz = torch.mean(kldiv_horiz, axis=4)
        kldiv_horiz[:, :, :, -1] = torch.ones_like(kldiv_horiz[:, :, :, -1]) * -1

        kldiv_vert = kldiv(input_preds, vert)
        kldiv_vert = torch.mean(kldiv_vert, axis=4)
        kldiv_vert[:, :, -1, :] = torch.ones_like(kldiv_vert[:, :, -1, :]) * -1

        kldiv_depth = kldiv(input_preds, depth)
        kldiv_depth = torch.mean(kldiv_depth, axis=4)
        kldiv_depth[:, -1, :, :] = torch.ones_like(kldiv_depth[:, -1, :, :]) * -1

        max_vert_horiz = torch.maximum(kldiv_horiz, kldiv_vert)
        kl_vals = torch.maximum(max_vert_horiz, kldiv_depth)

        # Use KL Divergence to find the PDBs
        threshold = torch.quantile(kl_vals, 0.99)
        pdb_indices = torch.nonzero((kl_vals >= threshold), as_tuple=True)
        pdb_index_tensor = torch.nonzero((kl_vals >= threshold))
        pdb_vals = input_preds[pdb_indices]

        # Find GDBs
        gt_horiz = torch.zeros(target.shape)
        gt_horiz[:, :, :, :-1] = target[:, :, :, 1:]

        gt_vert = torch.zeros(target.shape)
        gt_vert[:, :, :-1, :] = target[:, :, 1:, :]

        gt_depth = torch.zeros(target.shape)
        gt_depth[:, :-1, :, :] = target[:, 1:, :, :]

        gdb = (target * 3 != (gt_horiz + gt_vert + gt_depth))

        # Ensure no distance matching across volumes in batch
        gdb_dist = torch.from_numpy(distance_transform_edt((gdb == 0).cpu(), sampling=[99999999, 1, 1, 1])).to(device).float()

        # Compute the shape for d_pi - same as input_preds but last axis is 26 instead of channels
        d_pi_shape = (len(pdb_vals), len(self.directions))
        d_pi = torch.zeros(d_pi_shape)
        dists = torch.ones(d_pi_shape) * float('inf')

        # Compute d_pi and distances
        for m in range(len(self.directions)):
            direction = self.directions[m]
            pdb_offsets = torch.zeros_like(input_preds)
            dist_array = torch.zeros(input_preds.shape[:-1])
            s_start = torch.zeros(3, dtype=torch.int16)
            s_end = torch.zeros(3, dtype=torch.int16)
            i_start = torch.zeros(3, dtype=torch.int16)
            i_end = torch.zeros(3, dtype=torch.int16)
            zero = torch.ones(3, dtype=torch.int16)
            for i in range(3):
                if direction[i] == 1:
                    i_start[i] = 1
                    i_end[i] = input_preds.shape[i + 1]
                    s_start[i] = 0
                    s_end[i] = -1
                    zero[i] = input_preds.shape[i + 1] - 1
                elif direction[i] == -1:
                    i_start[i] = 0
                    i_end[i] = -1
                    s_start[i] = 1
                    s_end[i] = input_preds.shape[i + 1]
                    zero[i] = 0
                else:
                    i_start[i] = 0
                    i_end[i] = input_preds.shape[i + 1]
                    s_start[i] = 0
                    s_end[i] = input_preds.shape[i + 1]
            # This is our offset tensor
            pdb_offsets[:, s_start[0]:s_end[0], s_start[1]:s_end[1], s_start[2]:s_end[2], :] = input_preds[:, i_start[0]:i_end[0], i_start[1]:i_end[1], i_start[2]:i_end[2], :]
            pdb_offsets = pdb_offsets[pdb_indices]
            # Dists is just the offsetted gdb_dist
            dist_array[:, s_start[0]:s_end[0], s_start[1]:s_end[1], s_start[2]:s_end[2]] = gdb_dist[:, i_start[0]:i_end[0], i_start[1]:i_end[1], i_start[2]:i_end[2]]
            dists[:, m] = dist_array[pdb_indices]

            # Mean the elementwise KL Divergence first, then exponentiate
            kldiv_result = torch.exp(torch.mean(kldiv(pdb_vals, pdb_offsets.detach()), axis=-1))

            # Zero out anything that doesn't correspond to our offset - basically a bounds check
            if zero[0] != 1:
                kldiv_result = torch.where((pdb_index_tensor.T[0] == zero[0]), kldiv_result, torch.zeros(1))
            if zero[1] != 1:
                kldiv_result = torch.where((pdb_index_tensor.T[1] == zero[1]), kldiv_result, torch.zeros(1))
            if zero[2] != 1:
                kldiv_result = torch.where((pdb_index_tensor.T[2] == zero[2]), kldiv_result, torch.zeros(1))

            d_pi[:, m] = kldiv_result

        # Divide d_pi by the sum in the m axis
        d_pi /= torch.sum(d_pi, axis=-1).unsqueeze(axis=-1)

        # Compute d_gi
        d_gi = torch.ones_like(d_pi) * (0.2 / len(self.directions))
        min_indices = torch.argmin(dists, axis=-1, keepdim=True)
        d_gi = d_gi.scatter(axis=-1, index=min_indices, src=torch.ones_like(d_gi) * 0.8)

        # Cap weight at theta
        weight = torch.where(gdb_dist < self.theta, gdb_dist, torch.Tensor([self.theta])) / self.theta
        weight = weight[pdb_indices]

        # Only take loss where pdbs is 1 and gdb_dist is nonzero
        loss = weight * torch.mean(cross_entropy(d_pi, d_gi), axis=-1)
        loss = torch.where((gdb_dist[pdb_indices] != 0), loss, torch.zeros(1))

        # Calculate the total loss as the sum divided by the number of pdb pixels
        total_loss = torch.sum(loss)
        total_loss /= pdb_indices[0].shape[0]
        torch.set_default_tensor_type('torch.FloatTensor')
        return total_loss

class CE_IoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, input, target):
        return self.lovasz(input, target) + self.ce(input, target)

class CE_IABLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss()
        self.ce = nn.CrossEntropyLoss()
        self.abl = ActiveBoundaryLoss()
    def forward(self, input, target):
        return self.lovasz(input, target) + self.ce(input, target) + 0.2 * self.abl(input, target)

loss_mappings = {
    'test': TestLoss(),
    'abl': ActiveBoundaryLoss(),
    'cross_entropy': nn.CrossEntropyLoss(),
    'lovasz': LovaszSoftmaxLoss(),
    'ce+iou': CE_IoULoss(),
    'ce+iabl': CE_IABLoss(),
}
