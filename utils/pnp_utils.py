import cv2
import numpy as np
import torch

from utils.cpc import CPC_R


def check_iteration(iteration, prev_error, cur_error, jac, error, updates, params):
    if prev_error is None and cur_error is None:
        return True
    else:
        eps_1 = 1e-8
        eps_2 = 1e-8
        params = np.asarray(params) - updates.numpy()
        g = jac.T @ error.flatten()
        f = eps_2 * (np.linalg.norm(params) + eps_2)
        if torch.norm(g, p=np.inf).item() < eps_1:
            return False
        elif torch.norm(updates).item() < f:
            return False
        elif iteration > 50:
            return False
        else:
            return True


def check_lambda(lam, factor, prev_error, cur_error, jac, updates):
    if prev_error is None:
        return lam, factor
    prev_cost_function = 1/2 * (prev_error.flatten().T @ prev_error.flatten())
    cur_cost_function = 1/2 * (cur_error.flatten().T @ cur_error.flatten())
    gain_ratio = (prev_cost_function - cur_cost_function) / (1/2 * updates.T @ (lam * updates - jac.T @ cur_error.flatten()))
    if gain_ratio <= 0:
        lam = lam * factor
        factor = factor * 2
    else:
        lam = lam * np.max([1/3, 1 - (2 * gain_ratio - 1) ** 3])
        factor = 2

    return lam, factor


def cpc_rodr_4_angles(focals, centers, keypoints_pred, kpoints3D):
    cpc_pnp = CPC_R(focals, centers)

    rvects = []
    tvects = []
    errors = []

    kpoints3D_tensor = torch.from_numpy(kpoints3D).float()
    kpoints2D_tensor = torch.from_numpy(keypoints_pred).float()
    tvect = np.array([[0], [0], [10]])
    tvect_tensor = torch.from_numpy(tvect.reshape(-1)).float()

    # CPC Rodrigues 0° degrees
    rvect_0 = np.array([[1.1509305], [-1.1552572], [1.2745042]])
    rvect_tensor = torch.from_numpy(rvect_0.reshape(-1)).float()
    rvect_0, tvect_0, error_0 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, rvect_tensor,
                                        tvect_tensor, check_iteration, check_lambda)
    rvect_0 = rvect_0.numpy()
    tvect_0 = tvect_0.numpy()

    rvects.append(rvect_0)
    tvects.append(tvect_0)
    errors.append(error_0)

    # print(f'Reprojection error 0°: {error_0}')
    # print(f'Rvect 0°: {rvect_0}')

    # CPC Rodrigues 90° degrees
    rvect_90 = np.array([[-0.12036987], [2.4503145], [-2.0552557]])
    rvect_tensor = torch.from_numpy(rvect_90.reshape(-1)).float()
    rvect_90, tvect_90, error_90 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, rvect_tensor,
                                           tvect_tensor, check_iteration, check_lambda)
    rvect_90 = rvect_90.numpy()
    tvect_90 = tvect_90.numpy()

    rvects.append(rvect_90)
    tvects.append(tvect_90)
    errors.append(error_90)

    # print(f'Reprojection error 90°: {error_90}')
    # print(f'Rvect 90°: {rvect_90}')

    # CPC Rodrigues 180° degrees
    rvect_180 = np.asarray([[1.2133899], [1.1018114], [-1.120625]])
    rvect_tensor = torch.from_numpy(rvect_180.reshape(-1)).float()
    rvect_180, tvect_180, error_180 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, rvect_tensor,
                                              tvect_tensor, check_iteration, check_lambda)
    rvect_180 = rvect_180.numpy()
    tvect_180 = tvect_180.numpy()

    rvects.append(rvect_180)
    tvects.append(tvect_180)
    errors.append(error_180)

    # print(f'Reprojection error 180°: {error_180}')
    # print(f'Rvect 180°: {rvect_180}')

    # CPC Rodrigues 270° degrees
    rvect_270 = np.asarray([[1.6997603], [0.19744678], [-0.05384163]])
    rvect_tensor = torch.from_numpy(rvect_270.reshape(-1)).float()
    rvect_270, tvect_270, error_270 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, rvect_tensor,
                                              tvect_tensor, check_iteration, check_lambda)
    rvect_270 = rvect_270.numpy()
    tvect_270 = tvect_270.numpy()

    rvects.append(rvect_270)
    tvects.append(tvect_270)
    errors.append(error_270)

    # print(f'Reprojection error 270°: {error_270}')
    # print(f'Rvect 270°: {rvect_270}')

    rvects = np.asarray(rvects)
    tvects = np.asarray(tvects)
    errors = np.asarray(errors)
    min_error_idx = np.argmin(errors)

    rvect = rvects[min_error_idx][..., None]
    tvect = tvects[min_error_idx][..., None]

    # Flip rotation and translation matrices when result is flipped
    rmatr, _ = cv2.Rodrigues(rvect)
    rmatr[0] = rmatr[0] * np.sign(tvect[2, 0])
    rmatr[1] = rmatr[1] * np.sign(tvect[2, 0])
    rvect, _ = cv2.Rodrigues(rmatr)
    tvect = tvect * np.sign(tvect[2, 0])

    return errors[min_error_idx], rvect, tvect
