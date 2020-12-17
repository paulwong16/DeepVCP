import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation, query_ball_point, index_points


class DeepFeatureExtraction(nn.Module):
    def __init__(self, in_channel):
        super(DeepFeatureExtraction, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=4096, radius=0.1, nsample=32, in_channel=in_channel + 3, mlp=[32, 32],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=32 + 3, mlp=[32, 64],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=512, radius=0.4, nsample=32, in_channel=64 + 3, mlp=[64, 64],
                                          group_all=False)
        self.fp3 = PointNetFeaturePropagation(64 + 64, [64, 64])
        self.fp2 = PointNetFeaturePropagation(64 + 32, [32, 32])
        self.fp1 = PointNetFeaturePropagation(32, [32, 32, 32])
        self.fc = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.7)

    def forward(self, pcl):
        s0_points = pcl
        s0_xyz = pcl[:, :3, :]
        s1_xyz, s1_points = self.sa1(s0_xyz, s0_points)
        s2_xyz, s2_points = self.sa2(s1_xyz, s1_points)
        s3_xyz, s3_points = self.sa3(s2_xyz, s2_points)

        s2_points = self.fp3(s2_xyz, s3_xyz, s2_points, s3_points)
        s1_points = self.fp2(s1_xyz, s2_xyz, s1_points, s2_points)
        s0_points = self.fp1(s0_xyz, s1_xyz, None, s1_points)
        s0_points = self.fc(s0_points.permute(0, 2, 1))
        s0_points = self.dropout(s0_points)
        return s0_points


class WeightingLayer(nn.Module):
    def __init__(self, n_points, K=64):
        super(WeightingLayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(n_points),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(n_points),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(8, 1),
            nn.Softplus()
        )
        self.K = K

    def forward(self, x):
        batch_size = x.size(0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return torch.topk(x.view(batch_size, -1), self.K)


class DeepFeatureEmbedding(nn.Module):
    def __init__(self, K, radius=1):
        super(DeepFeatureEmbedding, self).__init__()
        self.K = K
        self.radius = radius

    def forward(self, source, source_intensity, keypoints):
        keypoints = keypoints[:, :, :3]
        xyz = source[:, :, :3]
        B, N, C = xyz.shape
        S = keypoints.size(1)
        idx = query_ball_point(self.radius, self.K, xyz, keypoints)
        torch.cuda.empty_cache()
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
        torch.cuda.empty_cache()
        grouped_xyz_norm = grouped_xyz - keypoints.view(B, S, 1, C)
        torch.cuda.empty_cache()
        grouped_points = index_points(source, idx)
        intensity = index_points(source_intensity, idx)
        new_points = torch.cat([grouped_xyz_norm, intensity, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
        return new_points


class DeepVCP(nn.Module):
    def __init__(self, N1, N2, TopK=64, K=32, grid_r=0.4, grid_s=0.4):
        super(DeepVCP, self).__init__()
        self.N1 = N1
        self.N2 = N2
        self.pointnet2 = DeepFeatureExtraction(4)
        self.wl = WeightingLayer(N1, TopK)
        self.dfe = DeepFeatureEmbedding(K)
        self.dfe_layer = nn.Sequential(
            nn.Linear(36, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.poolinglayer = nn.MaxPool2d((1, 32))
        self.voxel_num = int((grid_r / grid_s) + 1)
        self.grid_r = grid_r
        self.TopK = TopK
        self.K = K
        self.CNN = nn.Sequential(
            nn.Conv3d(16, 3, 1),
            nn.ReLU(),
            nn.Conv3d(4, 3, 1),
            nn.ReLU(),
            nn.Conv3d(1, 3, 1),
            nn.Softmax()
        )

    def forward(self, source, target, T_prev):
        batch_size = source.size(0)
        source_feature = self.pointnet2(source)
        target_feature = self.pointnet2(target)
        topk_indices = self.wl(source_feature).indices
        source = source.permute(0, 2, 1)
        keypoints = source.gather(1, topk_indices.unsqueeze(-1).repeat(1, 1, source.size(2)))
        deep_feature_source = self.dfe(source_feature, source[:, :, 3].unsqueeze(-1), keypoints)
        deep_feature_source = self.dfe_layer(deep_feature_source).permute(0, 3, 1, 2)
        deep_feature_source = self.poolinglayer(deep_feature_source).squeeze(-1).permute(0, 2, 1)
        target = target.permute(0, 2, 1)
        transformed_keypoints = torch.matmul(keypoints, T_prev)
        grid = torch.meshgrid([torch.linspace(-self.grid_r, self.grid_r, self.voxel_num),
                               torch.linspace(-self.grid_r, self.grid_r, self.voxel_num),
                               torch.linspace(-self.grid_r, self.grid_r, self.voxel_num)])
        grid_points = torch.stack(grid, 3).view(1, -1, 3).repeat(batch_size, 1, 1).cuda()
        candidate_corresponding_points = grid_points.unsqueeze(1).repeat(1, self.TopK, 1, 1)
        candidate_corresponding_points += transformed_keypoints.unsqueeze(2).repeat(1, 1, self.voxel_num ** 3, 1)[:, :, :, :3]
        candidate_corresponding_points = candidate_corresponding_points.view(batch_size, -1, 3)
        deep_feature_target = self.dfe(target_feature, target[:, :, 3].unsqueeze(-1), candidate_corresponding_points)
        deep_feature_target = self.dfe_layer(deep_feature_target)
        deep_feature_target = self.poolinglayer(deep_feature_target)
        deep_feature_target = deep_feature_target.view(batch_size, self.TopK, self.voxel_num ** 3, self.K, -1)

        return source


if __name__ == '__main__':
    model = DeepVCP(4096, 4096).cuda()
    pcl1 = torch.rand(1, 4, 4096)
    pcl2 = torch.rand(1, 4, 4096)
    T = torch.eye(4)
    res = model(pcl1.cuda(), pcl2.cuda(), T.cuda())
    print(res)
    # model = DeepFeatureEmbedding(32).cuda()
    # pcl = torch.rand(2, 4096, 32)
    # pcl2 = torch.rand(2, 64, 3)
    # intensity = torch.rand(2, 4096, 1)
    # model(pcl.cuda(), intensity.cuda(), pcl2.cuda())
    pass
