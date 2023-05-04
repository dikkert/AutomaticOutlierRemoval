# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:49:42 2022

@author: Linh
"""

import copy
import numpy as np
from sklearn.neighbors import KDTree
import open3d as o3d

class searching_neighbour3d:
    def searching_neighbour_points(src_pts_xyz: np.ndarray, trg_pts_xyz: np.ndarray, searching_neighbour: dict = {"searching_method": "knn", "searching_val": int(20)}):
        """
        Searching neighbour points by either knn or range search

        Parameters
        ----------
        src_pts_xyz : TYPE
            DESCRIPTION.
        trg_pts_xyz : TYPE
            DESCRIPTION.
        searching_neighbour : dict, optional
            DESCRIPTION. The default is {"searching_method": "knn", "searching_val": int(20)}.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        neighbour_pts_ids : TYPE
            DESCRIPTION.

        """
        # Generate kdtree
        leaf_size = 50
        tree = KDTree(src_pts_xyz, leaf_size = leaf_size)

        if searching_neighbour["searching_method"].casefold() == 'range'.casefold():
            # Range search
            searching_val = searching_neighbour['searching_val'] if searching_neighbour['searching_val'] > 0 else 0.1
            neighbour_pts_ids = tree.query_radius(trg_pts_xyz, r = searching_val)
        elif searching_neighbour["searching_method"].casefold() == 'knn'.casefold():
    		# kNN search
            searching_val = int(searching_neighbour['searching_val']) if int(searching_neighbour['searching_val']) > 5 else 20
            if searching_val >= src_pts_xyz.shape[0]: 
                searching_val = trg_pts_xyz.shape[0] - 1
            __, neighbour_pts_ids = tree.query(trg_pts_xyz, k=searching_val)
        else:
            raise ValueError('The searching method does not support')
        
        return neighbour_pts_ids
    
    def o3d_searching_neighour(src_pts_xyz, trg_pts_xyz, searching_neighbour: dict={"searching_method": "knn", "searching_val": 20}):
        """
        Searching neighbour by using open3D based on fklan

        Parameters
        ----------
        src_pts_xyz : the query points
        trg_pts_xyz : the data used to extract neighbour points of the query points
        searching_neighbour : dict, optional
            DESCRIPTION. The default is {"searching_method": "knn", "searching_val": 20}.

        Returns
        -------
        neighbour_pts_ids : TYPE
            DESCRIPTION.
            
        Demo:
            neighbour_pts_ids = searching_neighbour3d.o3d_searching_neighour(src_pts_xyz, trg_pts_xyz, searching_neighbour=searching_neighbour)
        """
        # Create tree for the target points
        trg_pts_xyz = copy.deepcopy(trg_pts_xyz)
        trg_pcd = o3d.geometry.PointCloud()
        trg_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(trg_pts_xyz))
        trg_tree = o3d.geometry.KDTreeFlann(trg_pcd)
        
        # Check dimensions of src_pt_xyz
        if src_pts_xyz.ndim == 1:
            src_pts_xyz = src_pts_xyz.reshape(-1, 3)
            
        # Searching 
        neighbour_pts_ids = []
        if searching_neighbour["searching_method"].casefold() == "hybrid".casefold():
            for src_pt_xyz in src_pts_xyz:
                [_, idx, _] = trg_tree.search_hybrid_vector_3d(src_pt_xyz, searching_neighbour["searching_val"][0], 
                                                               searching_neighbour["searching_val"][1])
                neighbour_pts_ids.append(np.asarray(idx))
            
        elif searching_neighbour["searching_method"].casefold() == "radius".casefold():
            for src_pt_xyz in src_pts_xyz:
                [_, idx, _] = trg_tree.search_radius_vector_3d(src_pt_xyz, searching_neighbour["searching_val"])
                neighbour_pts_ids.append(np.asarray(idx))
        elif searching_neighbour["searching_method"].casefold() == "knn".casefold():
            for src_pt_xyz in src_pts_xyz:
                [_, idx, _] = trg_tree.search_knn_vector_3d(src_pt_xyz, searching_neighbour["searching_val"])
                neighbour_pts_ids.append(np.asarray(idx))
                
        else:
            print("seaching method {:s} does not support and the radius searching with a radius of 0.1m is set".format(searching_neighbour["searching_method"]))
            for src_pt_xyz in src_pts_xyz:
                [_, idx, _] = trg_tree.search_radius_vector_3d(src_pt_xyz, 0.1)
                neighbour_pts_ids.append(np.asarray(idx))
        # return
        return neighbour_pts_ids   

    # def o3d_hybrid_searching(src_pts_xyz, trg_pts_xyz, hybrid_searching_neighbour: dict = {"knn": 50, "radius": 0.1}):
    #     # Create tree
    #     trg_pcd = o3d.geometry.PointCloud()
    #     trg_pcd.points = o3d.utility.Vector3dVector(trg_pts_xyz)
    #     trg_tree = o3d.geometry.KDTreeFlann(trg_pcd)
    #     # Searching 
    #     neighbour_pts_ids = []
    #     for src_pt_xyz in src_pts_xyz:
    #         [_, idx, _] = trg_tree.search_hybrid_vector_3d(src_pt_xyz, hybrid_searching_neighbour["radius"], 
    #                                                       hybrid_searching_neighbour["knn"])
    #         neighbour_pts_ids.append(np.asarray(idx))
    #     return neighbour_pts_ids
    
    # def o3d_radius_searching(src_pts_xyz, trg_pts_xyz, searching_radius: float=0.1):
    #     # Create tree
    #     trg_pcd = o3d.geometry.PointCloud()
    #     trg_pcd.points = o3d.utility.Vector3dVector(trg_pts_xyz)
    #     trg_tree = o3d.geometry.KDTreeFlann(trg_pcd)
    #     # Searching 
    #     neighbour_pts_ids = []
    #     for src_pt_xyz in src_pts_xyz:
    #         [_, idx, _] = trg_tree.search_radius_vector_3d(src_pt_xyz, searching_radius)
    #         neighbour_pts_ids.append(np.asarray(idx))
    #     return neighbour_pts_ids
    
    # def o3d_knn_searching(src_pts_xyz, trg_pts_xyz, searching_knn: int=50):
    #     # Create tree
    #     trg_pcd = o3d.geometry.PointCloud()
    #     trg_pcd.points = o3d.utility.Vector3dVector(trg_pts_xyz)
    #     trg_tree = o3d.geometry.KDTreeFlann(trg_pcd)
    #     # Searching 
    #     neighbour_pts_ids = []
    #     for src_pt_xyz in src_pts_xyz:
    #         [_, idx, _] = trg_tree.search_knn_vector_3d(src_pt_xyz, searching_knn)
    #         neighbour_pts_ids.append(np.asarray(idx))
    #     return neighbour_pts_ids