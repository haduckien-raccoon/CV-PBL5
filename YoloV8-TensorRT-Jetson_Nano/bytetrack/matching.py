import numpy as np
from scipy.optimize import linear_sum_assignment

def linear_assignment(cost_matrix, thresh):
    """Thay thế hoàn toàn thư viện lap"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches, unmatched_a, unmatched_b = [], [], []
    
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_a.append(r)
            unmatched_b.append(c)
        else:
            matches.append([r, c])
            
    unmatched_a.extend(list(set(range(cost_matrix.shape[0])) - set(row_ind)))
    unmatched_b.extend(list(set(range(cost_matrix.shape[1])) - set(col_ind)))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)
        
    return matches, tuple(unmatched_a), tuple(unmatched_b)


def ious(atlbrs, btlbrs):
    """Tính toán IoU bằng thuần NumPy, thay thế cython_bbox"""
    ious_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious_matrix.size == 0:
        return ious_matrix

    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)

    atlbrs_exp = atlbrs[:, np.newaxis, :]
    btlbrs_exp = btlbrs[np.newaxis, :, :]

    tl = np.maximum(atlbrs_exp[..., :2], btlbrs_exp[..., :2])
    br = np.minimum(atlbrs_exp[..., 2:], btlbrs_exp[..., 2:])

    wh = np.maximum(0.0, br - tl)
    area_i = wh[..., 0] * wh[..., 1]

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    area_u = area_a[:, np.newaxis] + area_b[np.newaxis, :] - area_i

    ious_matrix = area_i / np.maximum(area_u, 1e-6)
    return ious_matrix


def iou_distance(atracks, btracks):
    """Tính ma trận khoảng cách IoU cho ByteTrack"""
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks)>0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
        
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """Kết hợp score của YOLOv8 vào ma trận khoảng cách (Tùy chọn tăng độ chính xác)"""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
