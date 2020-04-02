import cv2
import numpy as np
import torch

from utils.cpc import CPC
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


def cpc_4_angles(focals, centers, kpoints2D, kpoints3D):
    cpc_pnp = CPC(focals, centers)

    alphas = []
    tvects = []
    errors = []

    kpoints3D_tensor = torch.from_numpy(kpoints3D).float()
    kpoints2D_tensor = torch.from_numpy(kpoints2D).float()
    tvect = np.array([[0], [0], [10]])
    tvect_tensor = torch.from_numpy(tvect.reshape(-1)).float()

    # CPC 0° angle
    alphas_tensor_0 = torch.FloatTensor([0, 0, 0])
    alphas_0, tvect_0, error_0 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, alphas_tensor_0,
                                         tvect_tensor, check_iteration, check_lambda)
    alphas_0 = alphas_0.numpy()
    tvect_0 = tvect_0.numpy()

    alphas.append(alphas_0)
    tvects.append(tvect_0)
    errors.append(error_0)

    print(f'Reprojection error 0°: {error_0}')
    print(f'Alphas 0°: {alphas_0}')

    # CPC 90° angle
    alphas_tensor_90 = torch.FloatTensor([0, np.pi / 4, 0])
    alphas_90, tvect_90, error_90 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor, alphas_tensor_90,
                                            tvect_tensor, check_iteration, check_lambda)
    alphas_90 = alphas_90.numpy()
    tvect_90 = tvect_90.numpy()

    alphas.append(alphas_90)
    tvects.append(tvect_90)
    errors.append(error_90)

    print(f'Reprojection error 90°: {error_90}')
    print(f'Alphas 90°: {alphas_90}')

    # CPC 180° angle
    alphas_tensor_180 = torch.FloatTensor([0, np.pi / 2, 0])
    alphas_180, tvect_180, error_180 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor,
                                               alphas_tensor_180, tvect_tensor, check_iteration,
                                               check_lambda)
    alphas_180 = alphas_180.numpy()
    tvect_180 = tvect_180.numpy()

    alphas.append(alphas_180)
    tvects.append(tvect_180)
    errors.append(error_180)

    print(f'Reprojection error 180°: {error_180}')
    print(f'Alphas 180°: {alphas_180}')

    # CPC 270° angle
    alphas_tensor_270 = torch.FloatTensor([0, (3 * np.pi) / 4, 0])
    alphas_270, tvect_270, error_270 = cpc_pnp(kpoints3D_tensor, kpoints2D_tensor,
                                               alphas_tensor_270, tvect_tensor, check_iteration,
                                               check_lambda)
    alphas_270 = alphas_270.numpy()
    tvect_270 = tvect_270.numpy()

    alphas.append(alphas_270)
    tvects.append(tvect_270)
    errors.append(error_270)

    print(f'Reprojection error 270°: {error_270}')
    print(f'Alphas 270°: {alphas_270}')

    alphas = np.asarray(alphas)
    tvects = np.asarray(tvects)
    errors = np.asarray(errors)
    min_error_idx = np.argmin(errors)

    return errors[min_error_idx], alphas[min_error_idx], tvects[min_error_idx]


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


def solve_PNP(K, dist, keypoints_pred, kpoints3D):
    keypoints_pred_ = np.ascontiguousarray(keypoints_pred[:, np.newaxis, :])
    kpoints3D_ = np.ascontiguousarray(kpoints3D[:, np.newaxis, :])
    _, rvect, tvect = cv2.solvePnP(kpoints3D_, keypoints_pred_, K, dist, cv2.SOLVEPNP_UPNP)

    # Flip rotation and translation matrices when result is flipped
    rmatr, _ = cv2.Rodrigues(rvect)
    rmatr[0] = rmatr[0] * np.sign(tvect[2, 0])
    rmatr[1] = rmatr[1] * np.sign(tvect[2, 0])
    rvect, _ = cv2.Rodrigues(rmatr)
    tvect = tvect * np.sign(tvect[2, 0])

    kpoints2D_pred, _ = cv2.projectPoints(kpoints3D_, rvect, tvect, K, dist)
    kpoints2D_pred = kpoints2D_pred.reshape(-1, 2)
    error = ((kpoints2D_pred - keypoints_pred) ** 2).mean()

    print(f'PnP reprojection error: {error}')
    print(f'PnP Rvect: {rvect}')

    return error, rvect, tvect
