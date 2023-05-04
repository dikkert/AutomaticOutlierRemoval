# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 21:06:53 2021

@author: LinhTruongHong
"""
import numpy as np
from geometry.vectors import vectors
from geometry.planes import planes

# Compare/searching
def inNd(a, b, assume_unique=False):

    # Check if a or b is not 1 dimension
    if a.ndim == 1:
        a = a.reshape(-1, len(a))

    if b.ndim == 1:
        b = b.reshape(-1, len(b))

    a = np.asarray(a, order='C')
    b = np.asarray(b, order='C')
    a = a.ravel().view((np.void, a.dtype.itemsize*a.shape[1]))
    b = b.ravel().view((np.void, b.dtype.itemsize*b.shape[1]))

    return np.in1d(a, b, assume_unique)


def get_block_points(pts_ids, pts_attr, tol_gap=0):
    """
    The function is to get the continous points defined by the maximum distance between two consecutive points
    Parameters
    ----------
    pts_ids             : indicies of points
    pts_attr            : values of points
    tol_gap             : gap tolerance. The default is 0.

    Returns
    -------
    blocks_region       : [N x 2] [ pts_ids, block_ids]
    
    pts_ids, pts_attr, tol_gap = peak_data_ids, peak_pts1d, gap_tol
    """
    # Sort data
    mask = np.argsort(pts_attr)
    pts_ids, pts_attr = pts_ids[mask], pts_attr[mask]
    
    # Get the gaps
    diff_pts_attr = np.diff(pts_attr)
    mask = diff_pts_attr > tol_gap
    if any(mask):
        gap_ids = np.where(mask == True)[0]
        s_block = np.unique(np.append(0, gap_ids + 1))
        e_block = np.unique(np.append(gap_ids, pts_ids.shape[0] - 1))
        block_ids = np.c_[s_block, e_block]
        # Assign region ids
        blocks_region_ids = np.full((pts_ids.shape[0], 2), np.inf, dtype=np.uint32)
        block_region_length = np.full((block_ids.shape[0], 2), np.inf, dtype=np.float32)
        for count, block_id in enumerate(block_ids):
            # Block_region_ids
            blocks_region_ids[block_id[0] : block_id[1] + 1, 0] = pts_ids[block_id[0] : block_id[1] + 1]
            blocks_region_ids[block_id[0] : block_id[1] + 1, 1] = count + 1
            # Block_region_length
            block_region_length[count, 0] = count + 1
            block_region_length[count, 1] = pts_attr[block_id[1]] - pts_attr[block_id[0]]
    else:
        blocks_region_ids = np.c_[pts_ids, np.full(pts_ids.shape[0], 1, dtype=np.uint32)].reshape(-1, 2)
        block_region_length = np.array([1, pts_attr[-1] - pts_attr[0]]).reshape(-1, 2)
        
    # Return
    mask = np.argsort(blocks_region_ids[:, 0])
    blocks_region_ids = blocks_region_ids[mask]
    return blocks_region_ids, block_region_length

def find_max(vals, tol: float= 0.0, max_consequent: int=3):
        """
        Find the largest values and its index  within the consequent values
    
        Parameters
        ----------
        vals : TYPE
            DESCRIPTION.
    
        Returns
        -------
        max_id : TYPE
            DESCRIPTION.
        max_val : TYPE
            DESCRIPTION.
    
        """
        # Constnt
        flag = True
        ind = max_id = 0
        max_val = vals[max_id]
        count_consequence = 0
        while flag & (ind < vals.shape[0] - 1):
            ind = ind + 1
            ind_val = vals[ind]
            if max_val + tol < ind_val:
                max_id = ind
                max_val = ind_val
                count_consequence = 0
            else:
                count_consequence += 1
                if count_consequence > max_consequent:
                    flag = False
        return max_id, max_val  



