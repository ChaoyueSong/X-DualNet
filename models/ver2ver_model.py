import torch
import models.networks as networks
import util.util as util
import numpy as np
import open3d as o3d
from models.networks.correspondence import warp

class Ver2VerModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

    def forward(self, identity_points, pose_points, id_face, arap_flag, mode):
        if mode == 'inference':
            pass
        else:
            identity_points=identity_points.cuda()
            pose_points=pose_points.cuda()

        generated_out = {}
        if mode == 'train':
            
            loss, generated_out = self.compute_loss(identity_points, pose_points, id_face, arap_flag)

            out = {}
            out['fake_points'] = generated_out['fake_points']
            out['identity_points'] = identity_points
            out['pose_points'] = pose_points
            out['fake_id_points'] = generated_out['fake_id_points']
            out['fake_pose_points'] = generated_out['fake_pose_points']
            return loss, out

        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(identity_points, pose_points)
            out['identity_points'] = identity_points
            out['pose_points'] = pose_points
            return out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr}]

        beta1, beta2 = opt.beta1, opt.beta2 
        G_lr = opt.lr

        optimizer = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)

        return optimizer

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netCorr'] = networks.define_Corr(opt)

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)

        return net

    def compute_loss(self, identity_points, pose_points, id_face, arap_flag):
        losses = {}
        generate_out = self.generate_fake(identity_points, pose_points, id_face, arap_flag)
        
        # edge loss
        losses['edge_loss'] = 0.0
        for i in range(len(identity_points)):  
            f = id_face[i].cpu().numpy()
            v = identity_points[i].cpu().numpy()
            losses['edge_loss'] = losses['edge_loss'] + util.compute_score(generate_out['fake_points'][i].transpose(1,0).unsqueeze(0),f,util.get_target(v,f,1))
        losses['edge_loss'] = losses['edge_loss']/len(identity_points) * self.opt.lambda_edge
        
        # dual reconstruction objective
        losses['rec_loss_id'] = torch.mean((generate_out['fake_id_points'] - identity_points.transpose(2,1))**2) * self.opt.lambda_rec
        losses['rec_loss_pose'] = torch.mean((generate_out['fake_pose_points'] - pose_points.transpose(2,1))**2) * self.opt.lambda_rec
         
        # backward correspondence loss
        losses['corr_bw_loss'] = generate_out['corr_bw_loss'] * self.opt.lambda_corr_bw

        return losses, generate_out

    def generate_fake(self, identity_points, pose_points, id_face, arap_flag):
        generate_out = {}
        
        bs = identity_points.shape[0]
        vertex_num = identity_points.shape[1]
        
        id_features = self.net['netCorr'](identity_points)
        pose_features = self.net['netCorr'](pose_points)
        
        #first transfer:id1_pose1+id2_pose2=id1'_pose2': include correspondence learning (in warp function) and pose transfer
        generate_out['warp_out'], generate_out['corr_bw_loss'] = warp(id_features, pose_features, pose_points, corr_bk_flag=True)
        generate_out['fake_points'] = self.net['netG'](id_features, generate_out['warp_out']) 
        
        #ARAP Deformer
        arap_num = int(vertex_num/10)  # choose 10% of vertices for arap
        if arap_flag:
            fake_points_list = []
            with torch.no_grad():
                for i in range(bs):
                    id_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(identity_points[i].cpu().numpy()), o3d.utility.Vector3iVector(id_face[i].cpu().numpy()))
                    fake_points_np = generate_out['fake_points'][i].transpose(1,0).cpu().detach().numpy()
                    ids = []
                    handle_ids = np.random.choice(vertex_num,size=arap_num,replace=False)
                    for i in range(arap_num):
                        ids.append(handle_ids[i])
                    constraint_ids = o3d.utility.IntVector(ids)
                    constraint_pos = o3d.utility.Vector3dVector(fake_points_np[handle_ids, :])

                    mesh_prime = id_mesh.deform_as_rigid_as_possible(constraint_ids,
                                                                constraint_pos,
                                                                max_iter=50)
                    fake_points_list.append(torch.from_numpy(np.asarray(mesh_prime.vertices).astype(np.float32)).transpose(1,0).cuda())
                generate_out['fake_points'] = torch.stack(fake_points_list, dim=0)
        
        #second transfer:id1'_pose2'+id1_pose1=id1'_pose1'
        fake_features = self.net['netCorr'](generate_out['fake_points'])
        warp_out_id = warp(fake_features, id_features, identity_points)
        generate_out['fake_id_points'] = self.net['netG'](fake_features, warp_out_id) 
        
        #third transfer:id2_pose2+id1'_pose2'=id2'_pose2'
        warp_out_pose = warp(pose_features, fake_features, generate_out['fake_points'])
        generate_out['fake_pose_points'] = self.net['netG'](pose_features, warp_out_pose) 

        return generate_out

    def inference(self, identity_points, pose_points):
        generate_out = {}
        
        id_features = self.net['netCorr'](identity_points)
        pose_features = self.net['netCorr'](pose_points)
        
        generate_out['warp_out'] = warp(id_features, pose_features, pose_points)
        generate_out['fake_points'] = self.net['netG'](id_features, generate_out['warp_out']) 

        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0