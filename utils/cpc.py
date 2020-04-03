import numpy as np
import torch
from torch.autograd import grad


# CamPoseCalib (CPC) method using Rodrigues rotation vector and translation vector
class CPC_R(torch.nn.Module):

    def apply_U(self, alpha_x, alpha_y, alpha_z):
        self.U[0, 1], self.U[1, 0] = -alpha_z, alpha_z
        self.U[0, 2], self.U[2, 0] = alpha_y, -alpha_y
        self.U[1, 2], self.U[2, 1] = -alpha_x, alpha_x

    def apply_r(self, alpha_x, alpha_y, alpha_z):
        self.r[0] = alpha_x
        self.r[1] = alpha_y
        self.r[2] = alpha_z

    def apply_tr(self, tr_x, tr_y, tr_z):
        self.Tr[0] = tr_x
        self.Tr[1] = tr_y
        self.Tr[2] = tr_z

    @staticmethod
    def compute_jacobian(jac_zero, outputs: torch.FloatTensor, inputs: list):
        num_outputs, output_features = outputs.shape
        num_params = len(inputs)
        assert jac_zero.shape == (num_outputs * output_features, num_params)

        for point_idx in range(len(inputs)):
            for coord_idx in range(output_features):
                grads = grad(outputs=outputs[point_idx, coord_idx],
                             inputs=inputs, retain_graph=True)
                for g_idx, g in enumerate(grads):
                    jac_zero[point_idx * output_features + coord_idx, g_idx] = grads[g_idx]

        return jac_zero

    def __init__(self, focals: np.ndarray, centers: np.ndarray):
        super(CPC_R, self).__init__()
        self.register_buffer('focals', torch.FloatTensor(focals))
        self.register_buffer('centers', torch.FloatTensor(centers))
        self.register_parameter('U', torch.nn.Parameter(torch.zeros(3, 3)))
        self.register_parameter('r', torch.nn.Parameter(torch.zeros(3)))
        self.register_parameter('Tr', torch.nn.Parameter(torch.zeros(3)))

    def forward(self, points3d: torch.FloatTensor, points2d: torch.FloatTensor,
                initial_rodrigues: torch.FloatTensor, initial_tr: torch.FloatTensor,
                policy_iterations: callable, policy_lambda: callable):

        assert len(points3d) == len(points2d)
        assert len(initial_rodrigues) == len(initial_tr) == 3

        n_points = len(points3d)
        n_params = len(initial_rodrigues) + len(initial_tr)

        # graph inputs that need grad
        rodr_x = torch.nn.Parameter(initial_rodrigues[0])
        rodr_y = torch.nn.Parameter(initial_rodrigues[1])
        rodr_z = torch.nn.Parameter(initial_rodrigues[2])

        tr_x = torch.nn.Parameter(initial_tr[0])
        tr_y = torch.nn.Parameter(initial_tr[1])
        tr_z = torch.nn.Parameter(initial_tr[2])

        jac = torch.zeros(n_points * 2, n_params).to(self.focals.device)

        prev_error = None
        cur_error = None
        iteration = 0
        lam = None
        factor = 2
        error = None
        updates = None

        while policy_iterations(iteration, prev_error, cur_error, jac, error, updates,
                                [rodr_x.item(), rodr_y.item(), rodr_z.item(),
                                 tr_x.item(), tr_y.item(), tr_z.item()]):

            self.apply_tr(tr_x, tr_y, tr_z)

            self.apply_r(rodr_x, rodr_y, rodr_z)

            theta = torch.norm(self.r)
            u = self.r / theta

            self.apply_U(u[0], u[1], u[2])
            u = u[..., None]
            rot_matrix = torch.eye(3).to(theta.device) * torch.cos(theta) + (
                        1. - torch.cos(theta)) * (u @ u.T) + self.U * torch.sin(theta)

            points_3d_camera = self.Tr + (rot_matrix @ points3d.T).T
            points_2d_pred = self.focals * points_3d_camera[:, :-1] / points_3d_camera[:, -1:] \
                             + self.centers

            error = points_2d_pred.detach() - points2d

            jac = self.compute_jacobian(jac, points_2d_pred,
                                        [rodr_x, rodr_y, rodr_z, tr_x, tr_y, tr_z])

            # =========== UPDATE STEP
            jac_dot = jac.T @ jac
            if jac_dot.sum() < 1e-7:
                break

            eye_matr = torch.eye(n_params).to(jac.device)

            if lam is None:
                eps = 1e-8
                lam = eps * torch.max(torch.diag(jac_dot)).item()

            try:
                updates = -torch.inverse((jac_dot + lam * eye_matr)) @ jac.T @ (error.flatten())
            except RuntimeError:
                break

            # this ensure detach from current graph
            rodr_x = torch.nn.Parameter(rodr_x.data.detach() + updates[0])
            rodr_y = torch.nn.Parameter(rodr_y.data.detach() + updates[1])
            rodr_z = torch.nn.Parameter(rodr_z.data.detach() + updates[2])

            tr_x = torch.nn.Parameter(tr_x.data.detach() + updates[3])
            tr_y = torch.nn.Parameter(tr_y.data.detach() + updates[4])
            tr_z = torch.nn.Parameter(tr_z.data.detach() + updates[5])

            prev_error = cur_error
            cur_error = error
            iteration += 1

            # =========== UPDATE LAMBDA PARAMETER
            lam, factor = policy_lambda(lam, factor, prev_error, cur_error, jac, updates)

        rt = torch.stack([rodr_x, rodr_y, rodr_z]).detach()
        tr = torch.stack([tr_x, tr_y, tr_z]).detach()
        return rt, tr, (error ** 2).mean().item()
