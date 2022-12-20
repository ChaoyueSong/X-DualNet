import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
import util.util as util
from .graph import Graph
from .gconv import SetConv


class FeatureExtractor(BaseNetwork):

    def __init__(self, opt):
        """
        adapted from https://github.com/valeoai/FLOT/blob/master/flot/models/scene_flow.py
        """
        self.opt = opt
        super().__init__()

        n = 32
        # Feature extraction
        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n) 

    def get_features(self, points, nb_neighbors):
        """
        Compute deep features for each point of the input point sets. These
        features are used to compute the transport cost matrix between two
        point sets.
        
        Parameters
        ----------
        points : torch.Tensor
            Input point set of size B x N x 3
        nb_neighbors : int
            Number of nearest neighbors for each point.

        Returns
        -------
        x : torch.Tensor
            Deep features for each point. Size B x N x 128
        graph : networks.graph.Graph
            Graph build on input point set containing list of nearest 
            neighbors (NN) and edge features (relative coordinates with NN).

        """

        graph = Graph.construct_graph(points, nb_neighbors)
        x = self.feat_conv1(points, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)

        return x, graph

    def forward(self, points):
        if points.shape[2] == 3:
            pass
        else:
            points = points.transpose(2,1)
        features, graph = self.get_features(points, 32)

        return features


def warp(id_features, pose_features, pose_points, corr_bk_flag=False):
    """
    Learn the correspondence between identity mesh and pose mesh via optimal transport,
    and warp the pose_points with the learned matching matrix.
    
    Parameters
    ----------
    id_features : torch.Tensor
        features of identity points: size B x N_id x 128
    pose_features : torch.Tensor
        features of pose points: size B x N_pose x 128
    corr_bk_flag : Bool
        whether to use backward correspondence loss

    Returns
    -------
    warp_out : torch.Tensor
        warped pose points : size B x N_pose x 3
    corr_bw_loss : torch.Tensor
        backward correspondence loss
    """

    # Correlation matrix C (cosine similarity)
    id_features_norm = torch.norm(id_features, 2, -1, keepdim=True) + sys.float_info.epsilon
    id_features = torch.div(id_features, id_features_norm)

    pose_features_norm = torch.norm(pose_features, 2, -1, keepdim=True) + sys.float_info.epsilon
    pose_features = torch.div(pose_features, pose_features_norm)
    pose_features_permute = pose_features.permute(0, 2, 1)

    C_Matrix = torch.matmul(id_features, pose_features_permute) 

    K = torch.exp(-(1.0 - C_Matrix) / 0.03)

    # Init. of Sinkhorn algorithm
    power = 1#gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=id_features.device, dtype=id_features.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=id_features.device, dtype=id_features.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=pose_features.device, dtype=pose_features.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(5):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Optimal matching matrix Tm
    T_m = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    T_m_norm = T_m /(torch.sum(T_m, dim=2, keepdim=True)+1e-8)
    
    if pose_points.shape[2] == 3:
        pass
    else:
        pose_points = pose_points.transpose(2,1)
    # Warped pose points
    warp_out = torch.matmul(T_m_norm, pose_points)
    
    # Backward correspondence loss
    if corr_bk_flag:
        warp_pose_back = torch.matmul(T_m_norm.transpose(2,1), warp_out)  # (bs, n_pose, 3)
        corr_bw_loss = torch.mean((warp_pose_back - pose_points)**2)
        return warp_out, corr_bw_loss
    else:
        return warp_out
