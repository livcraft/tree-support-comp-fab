from intersection import single_ray_mesh_intersection, parallel_ray_mesh_intersection

from trimesh.transformations import random_rotation_matrix
from copy import deepcopy

from io import TextIOWrapper
from typing import List

import os
import trimesh
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D

def write_triangle(f: TextIOWrapper, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                   normal: List[float]):
    '''
    Write a triangle facet to the output file stream 'f'.
    '''
    f.write(
        f"  facet normal {' '.join([f'{val:.6g}' for val in normal])}\n"
        f"    outer loop\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p1])}\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p2])}\n"
        f"      vertex {' '.join([f'{val:.6g}' for val in p3])}\n"
        f"    endloop\n"
        f"  endfacet\n"
    )


class Voxelizer:
    '''
    Voxelizer class.
    '''

    def __init__(self, stl_file_path: str, voxel_size: float):
        '''
        Constructor of the Voxelizer class.
        '''
        # Extract mesh name from the STL file path
        mesh_name = os.path.splitext(os.path.split(stl_file_path)[1])[0]

        # Read input mesh in STL file
        mesh = trimesh.load(stl_file_path)

        # Save the local variables as class members so that they can be used in other functions
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.voxel_size = voxel_size

        # Initialize the voxel grid
        self.init_voxels()


    def init_voxels(self):
        '''
        Initialize the voxel grid.
        '''
        # Compute the bounding box of the mesh
        # 'bbox_min' is the bottom left corner of the mesh, while 'bbox_max' is the top right corner
        bbox_min = self.mesh.vertices.min(axis=0)
        bbox_max = self.mesh.vertices.max(axis=0)

        # Allocate a voxel grid slightly bigger than the mesh by padding the bounding box of the mesh
        dx = self.voxel_size
        voxel_grid_min = bbox_min - dx
        voxel_grid_max = bbox_max + dx

        # Compute the number of voxels needed in each dimension
        # np.ceil() rounds the elements in an array up to the smallest integers
        # The results are then cast to integer data type
        voxel_grid_size = np.ceil((voxel_grid_max - voxel_grid_min) / dx)
        voxel_grid_size = voxel_grid_size.astype(np.int64)

        # Allocate an empty voxel grid
        # (using 8-bit unsigned integers saves memory since each voxel stores either 0 or 1)
        voxels = np.zeros(voxel_grid_size, dtype=np.uint8)

        # Save the grid info as class members
        self.voxel_grid_min = voxel_grid_min
        self.voxel_grid_size = voxel_grid_size
        self.voxels = voxels


    def get_support_points(self, bottom_intersections):
        """
        given a list of the bottom layer of an object, returns
        the list of support points for overhangs
        """
        support_pts = {}

        for pt,y in bottom_intersections.items():
            x_corners = []
            z_corners = []

            for x_val in [pt[0] - 1, pt[0] + 1]:
                if (x_val,pt[1]) in bottom_intersections :
                    x_corners.append((x_val, bottom_intersections[(x_val, pt[1])]))
            for z_val in [pt[1] - 1, pt[1] + 1]:
                if (pt[0], z_val) in bottom_intersections:
                    z_corners.append((z_val, bottom_intersections[(pt[0], z_val)]))

            if len(x_corners) == 2:
                x_ang = np.rad2deg(np.arctan2((x_corners[0][1] - x_corners[1][1]), (x_corners[0][0] - x_corners[1][0])))
            else:
                x_ang = np.rad2deg(np.arctan2((y - x_corners[0][1]), (pt[0] - x_corners[0][0])))
            
            if len(z_corners) == 2:
                z_ang = np.rad2deg(np.arctan2((z_corners[0][1] - z_corners[1][1]), (z_corners[0][0] - z_corners[1][0])))
            else:
                z_ang = np.rad2deg(np.arctan2((y - z_corners[0][1]), (pt[1] - z_corners[0][0])))

            if (abs(x_ang) > 145) and (abs(z_ang) > 145):
                new_y = y-1
                if new_y > 1:

                    support_pts[pt] = new_y

        # [x, y, z, label]
        model = self.clustering(support_pts)
        print(model)

        return support_pts

    def clustering(self, points):
        data = []

        for pt, y in points.items():
            data.append([pt[0], y, pt[1]])

        model = DBSCAN(eps=2.5, min_samples=2)
        model.fit_predict(np.array(data))
        labels = model.labels_
        num_labels  = len(set(labels))
        
        samples_w_lbls = np.concatenate((data,labels[:,np.newaxis]),axis=1)

        return samples_w_lbls




    def run_brute_force(self) -> float:
        '''
        Run brute-force voxelization.
        '''
        # Read voxel grid dimensions from the class
        nx, ny, nz = self.voxel_grid_size

        # These class members are frequently used so it's best to assign them to local variables
        mesh = self.mesh        # Input mesh
        dx = self.voxel_size    # Voxel size
        voxels = self.voxels    # Voxel grid

        # Compute the center of the bottom-left voxel, which we use to derive the centers of
        # other voxels
        voxel_bottom_left = self.voxel_grid_min + dx * 0.5
    
        bottom_origins = []
        bottom_intersections = {}
        # Loop over all positions in the voxel grid
        # Note that this nested loop is slow and might run for several minutes
        for x in range(nx):   
            for y in range(ny):       
                for z in range(nz):   

                    # Set ray origin as the current voxel center
                    ray_origin = np.array([(x*dx) + voxel_bottom_left[0], (y*dx) + voxel_bottom_left[1], (z*dx) + voxel_bottom_left[2]])
                    
                    # Set ray direction as positive X direction by default
                    ray_direction = np.array([0.0, 1.0, 0.0])

                    # Intersect the ray with the mesh and get the intersection locations
                    locations = single_ray_mesh_intersection(mesh, ray_origin, ray_direction)

                    if ray_origin[1] <= -4.98:
                        bottom_origins.append(ray_origin)
                        if locations:
                            y_val = int(min(locations) / self.voxel_size)
                            bottom_intersections[(x, z)] = y_val
                            
                    # Determine whether the voxel at the current grid point is inside the mesh.
                    # Recall from lectures that an odd number of intersections means inside
                    # if len(locations) % 2 == 0:
                    #     voxels[x, y, z] = 0
                    # else:
                    #     voxels[x, y, z] = 1

            print(f'Completed layer {x + 1} / {nx}')

        # use bottom_intersections to get support points
        support_points = self.get_support_points(bottom_intersections)
        #print(support_points)
        for pt,y in support_points.items():
            voxels[pt[0], y, pt[1]] = 1

        #generate voxels for support points

        #make this a loop while support points still exist

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(voxels) / voxels.size
        return occupancy


    def run_accelerated(self, check_result: bool=True) -> float:
        '''
        Run accelerated voxelization.
        '''
        # Read voxel grid dimensions from the class
        nx, ny, _ = self.voxel_grid_size

        # These class members are frequently used so it's best to assign them to local variables
        mesh = self.mesh                # Input mesh
        dx = self.voxel_size            # Voxel size
        voxels = self.voxels            # Voxel grid

        # Compute the origin of the bottom-left ray, which we use to derive other ray origins
        # Note that all ray origins lie on the Z=0 plane so they will be outside the mesh
        origin_bottom_left = self.voxel_grid_min + np.array([1, 1, 0]) * (dx * 0.5)

        # Precompute ray origins and directions
        num_rays = nx * ny
        ray_origins = origin_bottom_left + dx * \
            np.hstack((
                np.stack(np.mgrid[:nx, :ny], axis=2).reshape(-1, 2),
                np.zeros((num_rays, 1))
            ))
        ray_direction = np.array([0.0, 0.0, 1.0])

        # Clear the voxel grid
        voxels[:] = 0

        # Intersect the rays with the mesh
        intersections = parallel_ray_mesh_intersection(mesh, ray_origins, ray_direction, origins_outside=check_result)

        # Fill the voxels by looping over all rays
        for x in range(nx):      
            for y in range(ny):  

                # Get the intersections of the current ray
                distances = np.array(intersections[x * ny + y])

                # Only process rays with intersections
                if len(distances):

                    # Convert distances to alternate indices of interval endpoints
                    lower_indices = np.ceil(distances[::2] / dx - 0.5 - 1e-8).astype(np.int64)
                    upper_indices = np.floor(distances[1::2] / dx + 0.5 + 1e-8).astype(np.int64)

                    # Fill voxels within the interval
                    for l,u in zip(lower_indices, upper_indices):  
                        np.put(voxels[x, y], np.arange(l, u), 1) 

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(voxels) / voxels.size

        return occupancy


    def run_approximate(self, num_samples: int=20) -> float:
        '''
        Run approximate voxelization on a non-watertight mesh. This method actually doesn't check
        watertight-ness so it can run on any mesh.
        '''
        # Maximum number of samples is 255, otherwise the voxel counters will overflow
        if num_samples > 255:
            raise ValueError('At most 255 samples are supported')

        # Constants
        dx = self.voxel_size

        # Back up the current mesh
        mesh_backup = deepcopy(self.mesh)

        # Read the current voxel grid info
        nx, ny, nz = self.voxel_grid_size
        voxel_grid_size = self.voxel_grid_size
        grid_bottom_left = self.voxel_grid_min

        # Precompute the voxel centers in the current voxel grid
        voxel_centers = grid_bottom_left + dx * 0.5 + \
            np.stack(np.mgrid[:nx, :ny, :nz], axis=3).reshape(-1, 3) * dx

        # Collect an initial sample by running accelerated voxelization
        self.run_accelerated(check_result=False)
        voxels_count = self.voxels

        print(f'Finished 1 / {num_samples} samples')

        # Collect other samples by voxelizing the mesh in random directions
        for i in range(2, num_samples + 1):

            # Create a copy of the original mesh for us to work on
            self.mesh = deepcopy(mesh_backup)

            # Rotate the mesh using a random rotation matrix (with a size of 4x4)
            R = random_rotation_matrix()
            # --------
            # TODO: Your code here. Rotate the mesh by calling the `apply_transform` method of the
            # Trimesh class, defined in trimesh/base.py which is accessible on Github:
            # https://github.com/mikedh/trimesh/blob/master/trimesh/base.py
            self.mesh.apply_transform(R)

            # Voxelize the rotated mesh to obtain a new voxel grid
            self.init_voxels()
            # --------
            # TODO: Your code here. Run accelerated voxelization on the rotated mesh.
            # Note that you should set a proper value for keyword argument(s).
            self.run_accelerated(check_result=False)

            # Now we will sample the new voxel grid at the rotated positions of voxel centers
            # in the original voxel grid. First, we compute the rotated coordinates of the original
            # voxel centers
            # --------
            # TODO: Your code here. Rotate the voxel centers from the original coordinates
            # to the new coordinates (where the rotated mesh resides) using matrix multiplication.
            # In NumPy, matrix multiplication can be written as `np.dot(A, B)` where A is MxN and
            # B is NxP. Be aware that in this case `voxel_centers` is an Nx3 array while
            # the rotation matrix is 4x4. What else should be done?

            # rotated_voxel_centers = np.dot(np.c_[voxel_centers,np.zeros(len(voxel_centers))], R)[:,:3]
            rotated_voxel_centers = np.dot(voxel_centers, np.linalg.inv(R)[:3,:3])

            # Then, align the rotated voxel centers with the new bottom-left corner
            rotated_voxel_centers -= self.voxel_grid_min

            # Discard voxel centers outside the new voxel grid area
            new_voxel_grid_bound = self.voxel_grid_size * dx
            in_bound_mask = \
                np.all((rotated_voxel_centers >= 0) & \
                       (rotated_voxel_centers < new_voxel_grid_bound), axis=1)
            rotated_voxel_centers = rotated_voxel_centers[in_bound_mask]

            # Round in-bound voxel centers to integer coordinates
            rotated_indices = np.floor(rotated_voxel_centers / dx + 1e-8)
            rotated_indices = rotated_indices.astype(np.int64)

            # Now we extract the sampled values from the new voxel grid
            rid = rotated_indices
            new_voxels = self.voxels[rid[:, 0], rid[:, 1], rid[:, 2]]

            # Add the sampled values to the corresponding voxel counters
            voxels_count.ravel()[in_bound_mask] += new_voxels

            print(f'Finished {i} / {num_samples} samples')

        # Reset mesh and voxel grid info
        self.mesh = mesh_backup
        self.voxel_grid_size = voxel_grid_size
        self.voxel_grid_min = grid_bottom_left

        # Set grid occupany according to voxel counters
        # --------
        # TODO: Your code here. Assign the voxel grid with values according to the majority rule.
        # Please cast the result to `np.uint8` data type according to the default setting in
        # `init_voxels`.
    
        self.voxels = np.rint(voxels_count/num_samples).astype(np.uint8)

        # Compute the occupancy of the voxel grid, i.e., the fraction of voxels inside the mesh
        occupancy = np.count_nonzero(self.voxels) / self.voxels.size

        return occupancy


    def save_mesh(self, output_file_path: str):
        '''
        Save the voxel grid as a triangle mesh in STL format.
        '''
        # Read relevant variables from the class
        nx, ny, nz = self.voxel_grid_size
        dx = self.voxel_size
        voxels = self.voxels
        grid_bottom_left = self.voxel_grid_min

        # Precompute all grid point coordinates. Unlike voxel centers, grid points are offset by
        # half of the voxel size.
        grid_indices = np.mgrid[:nx + 1, :ny + 1, :nz + 1]
        grid_indices = np.stack(grid_indices, axis=3)
        grid_points = grid_bottom_left + dx * grid_indices

        # Cache all possible normals
        normals = np.hstack((-np.eye(3), np.eye(3))).reshape(-1, 3)

        # Cache all index slices
        slices = np.array([slice(None, -1), slice(1, None), slice(1, -1)])

        # Start writing to the output file
        with open(output_file_path, 'w') as f:
            # Write the header
            f.write('solid vcg\n')

            # Generate triangles perpendicular to X, Y, and Z direction respectively
            for dim, axis in enumerate('XYZ'):

                # Take the difference between neighboring voxels along the current axis
                # We need to generate a square facet whenever the difference is not zero,
                # which indicates voxelized mesh boundary
                diff = np.diff(voxels, axis=dim)

                # Consider two normal directions along each axis: positive and negative
                # Inside the inner loop, we generate the group of triangles that share the same
                # normal direction
                for positive in (0, 1):

                    # Compute the set of grid points that will be used as triangle vertices
                    grid_mask = diff == 0xff if positive else diff == 1
                    vertices = [
                        grid_points[
                            tuple(slices[np.roll([i // 2, i % 2, 2], dim + 1)].tolist())
                        ][grid_mask] for i in range(4)
                    ]

                    # Get the current normal direction
                    normal = normals[dim * 2 + positive]

                    # According to STL format, the order of vertices in each triangle satisfies the
                    # right-hand rule.
                    for p1, p2, p3, p4 in zip(*vertices):
                        if positive:
                            write_triangle(f, p1, p4, p2, normal)
                            write_triangle(f, p1, p3, p4, normal)
                        else:
                            write_triangle(f, p1, p2, p4, normal)
                            write_triangle(f, p1, p4, p3, normal)

            # Write the footer
            f.write('endsolid\n')

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")


    def save_to_txt_file(self, output_file_path: str):
        '''
        Save the voxelized mesh as a text file of 0-1 strings. This format is intended for
        user inspection.
        '''
        # Read relevant variables from the class
        nx, ny, nz = self.voxel_grid_size
        dx = self.voxel_size
        voxels = self.voxels
        grid_bottom_left = self.voxel_grid_min

        # Open/Create the output file in write mode
        with open(output_file_path, 'w') as f:
            # Write the bottom left position, voxel size, and grid dimensions
            f.write(f'{grid_bottom_left[0]} {grid_bottom_left[1]} {grid_bottom_left[2]} '
                    f'{dx} {nx - 1} {ny - 1} {nz - 1}\n')

            # Write the voxel grid
            for i in range(nx):
                for j in range(ny):
                    line_str = ''.join(voxels[i, j].astype('<U1'))
                    f.write(f'{line_str}\n')

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")


    def load_from_data_file(self, data_file_path: str):
        '''
        Load the voxel grid info from a binary archive file.
        '''
        # Load info from a numpy archive file, including bottom left position, voxel size,
        # voxel grid dimensions, and the voxel grid content
        data = np.load(data_file_path)

        self.voxel_grid_min = data['voxel_grid_min']
        self.voxel_size = data['voxel_size']
        self.voxel_grid_size = data['voxel_grid_size']
        self.voxels[:] = data['voxels']


    def save_to_data_file(self, output_file_path: str):
        '''
        Save the voxelized mesh as a binary archive file. This format is faster to load and
        preferred for grading.
        '''
        # Save info to a numpy archive file, including bottom left position, voxel size,
        # voxel grid dimensions, and the voxel grid content
        np.savez(
            output_file_path,
            voxel_grid_min=self.voxel_grid_min,
            voxel_size=self.voxel_size,
            voxel_grid_size=self.voxel_grid_size,
            voxels=self.voxels
        )

        # Display a finish message
        print(f"Voxelized mesh written to file '{output_file_path}'")
