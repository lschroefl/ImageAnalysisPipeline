## import some libraries
from skimage.measure import regionprops_table
import pandas as pd
import trackpy as tp
import numpy as np
from skimage.draw import disk
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
import cv2
from qtpy.QtWidgets import QFileDialog
import tifffile as tiff
import os
import ast
from natsort import natsorted
from scipy.ndimage import label, grey_dilation
from collections import defaultdict


## define the functions that are needed
def load_data(viewer, global_plugin_data, data_type = 'label', file_path_default = "", message = "Load some data"):
    """Load data from disk into the napari viewer. data_type either 'label', 'image' or 'table'"""
    if data_type == 'label': 
        print(message)
        file_path, _ = QFileDialog.getOpenFileName(None, "Load an image mask/label file", file_path_default, "Image Files (*.tif *.npy *.tiff)")
        if not file_path:
            print(message)
            return None
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
            print(f"You have loaded {file_path}")
            print(f"Shape: {data.shape}")
            print(f"Type: {type(data)}")
            print(f"Dtype: {data.dtype}")
            viewer.add_labels(data, name=os.path.basename(file_path))
        elif file_path.endswith((".tif", ".tiff")): ## Bug: tifs from alex get fucked when loaded uint8 issue I guess
            data = tiff.imread(file_path)
            print(f"You have loaded {file_path}")
            print(f"Shape: {data.shape}")
            print(f"Type: {type(data)}")
            print(f"Dtype: {data.dtype}")
            label_layer = viewer.add_labels(data, name=os.path.basename(file_path))
            return label_layer, data, file_path
        else:
            raise ValueError("Unsupported file format. Please select a .npy or .tif or .tsv file.")
    elif data_type == 'table':
        print(message)
        file_path, _ = QFileDialog.getOpenFileName(None, "Load a quantification table", "", "Matrix Files (*.tsv)")
        if not file_path:
            print("No file selected.")
            return None
        elif file_path.endswith((".tsv")):
            data = pd.read_csv(file_path, sep = '\t',  converters={"neighbor_id": ast.literal_eval, "neighbor_anno": ast.literal_eval}, dtype = {'frame' : 'int64', 'annotation' : 'object', 'particle' : 'int64'})
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            global_plugin_data[base_name] = data
            global_plugin_data[f'{base_name}_path'] = file_path        
            print(f"You have loaded {file_path}")
            print(f"Shape: {data.shape}")
            print(f"Type: {type(data)}")
            print(f"Table added to napari as global_plugin_data[{base_name}]")
            return data, file_path
        else:
            raise ValueError("Unsupported file format. Please select a .npy or .tif or .tsv file.")


## define the functions that are needed
def save_data(viewer, global_plugin_data, data, data_type = 'label', file_path_default = None, message = "Saving some incredible data"):
    """Save data to disk. data_type either 'label', 'image' or 'table'"""
    if (data_type == 'label') or (data_type == 'image'): 
        print(message)
        file_path, _ = QFileDialog.getSaveFileName(None, message, file_path_default, "TIF files (*.tif *.tiff *.npy)")
        if file_path:
            if not file_path.lower().endswith('.tif'):
                file_path += '.tif'
            tiff.imwrite(file_path, data)
            print(f"Image/mask saved successfully to {file_path}")
            return file_path
        elif not file_path:
            print("No file path selected, save operation aborted.")
            return None
    elif data_type == 'table':
        print(message)
        file_path, _ = QFileDialog.getSaveFileName(None, message, file_path_default, "Tables (*.tsv *.xlsx)")
        if file_path and isinstance(data, pd.DataFrame):
            data = data[natsorted(data.columns)]
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.tsv']:
                file_path = os.path.splitext(file_path)[0] + '.tsv'
            data.to_csv(file_path, sep = '\t', index = False) 
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            global_plugin_data[base_name] = data
            global_plugin_data[f'{base_name}_path'] = file_path      
            print(f"Table saved successfully to {file_path}")
            print(f"Table added to napari as global_plugin_data[{base_name}]") 
            return file_path
        elif file_path and isinstance(data, pd.core.groupby.generic.DataFrameGroupBy): 
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in ['.xlsx']:
                file_path = os.path.splitext(file_path)[0] + '.xlsx'
            group_dict = {group: table for group, table in data}
            print(group_dict.keys())
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for group in natsorted(group_dict.keys()):
                    print(f'{group} ')
                    sheet_name = str(group)[:31]  # Excel sheet name limit
                    group_dict[group] = group_dict[group][natsorted(group_dict[group].columns)]
                    group_dict[group].to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Saved grouped custom table to {file_path} with {len(group_dict)} sheets.")
        elif not file_path:
            print("No file path selected, save operation aborted.")
            return None
        else: 
            print("Not sure what went wrong, maybe the data type is neither a pd.Dataframe nor a pd.groupby object")
            return None



def centroids_and_basic_quantification(mask): 
    if mask is None:
        print("No mask data provided!")
        return(None)
    else:
        print(f'Mask with dimensions: {mask.shape} successfully loaded!')
        quantification_table = []
        # iterate over frames correctly (time is the first dimension)
        for timepoint, frame in enumerate(mask):
            properties = regionprops_table(
                frame,
                properties=[
                    'label',
                    'area',
                    'area_convex',
                    'perimeter',
                    'eccentricity',
                    'orientation',
                    'centroid'])
            for i in range(len(properties['label'])):
                quantification_table.append({
                    'frame': timepoint,
                    'area': properties['area'][i],
                    'area_convex': properties['area_convex'][i],  # corrected duplicate key
                    'perimeter': properties['perimeter'][i],
                    'eccentricity': properties['eccentricity'][i],
                    'orientation': properties['orientation'][i],
                    'centroid_x': properties['centroid-1'][i], 
                    'centroid_y': properties['centroid-0'][i], 
                    'initial_id': properties['label'][i],
                    'annotation': 'other'
                })
        # Create a DataFrame from quantification_table
        quantification_table = pd.DataFrame(quantification_table)
        return quantification_table


def link_centroids_relabel_mask(mask, quantification_table, search_range = 20, memory = 4, t_column = 'frame'):
    print('Tracking centroids over time and relabeling the mask')
    quantification_table = tp.link(quantification_table, 
                          pos_columns = ['centroid_y', 'centroid_x'], 
                          t_column = t_column,
                          search_range=search_range,  # Adjust search_range based on expected movement
                          memory = memory)
    ## increase all particle numbers by +1 to avoid issues with a particel/cell being labeled as zero (background)
    quantification_table['particle'] += 1
    ## relabel the segementation mask according to the particle id
    # Initialize the new stack
    mask_relabeled = []
    # Group tables by frame
    grouped_tables = quantification_table.groupby('frame')
    # Iterate over each frame and in mask
    for t, frame in enumerate(mask):
        new_frame = np.zeros_like(frame, dtype=np.int32)  # Initialize a new frame with zeros
        if t in grouped_tables.groups:
            # Get the tables for this frame
            frame_props = grouped_tables.get_group(t)
            # Iterate over each row in tables for this frame
            for _, row in frame_props.iterrows():
                initial_id = row['initial_id']
                particle_id = row['particle']
                # Replace regions with the particle number
                new_frame[frame == initial_id] = particle_id
        # Append the updated frame to the new stack
        mask_relabeled.append(new_frame)

    mask_relabeled = np.array(mask_relabeled)
    ## small test----------------------------------------
    print(f'New mask shape: {mask_relabeled.shape}, type: {type(mask_relabeled)}, dtype {mask_relabeled.dtype}')
    print(quantification_table.head()) 

    ## return the results
    return quantification_table, mask_relabeled


def convert_to_uint8(stack):
    new_stack = []
    for i, frame in enumerate(stack):
        """Normalize and convert any dtype frame to uint8."""
        frame = frame.astype(np.float32)  # Convert to float for safe normalization
        min_val, max_val = frame.min(), frame.max() 
        
        if max_val > min_val:  # Avoid division by zero
            frame = (frame - min_val) / (max_val - min_val) * 255  # Normalize
            new_stack.append(frame.astype(np.uint8))
        else: 
            raise ValueError(f'Frame {i} has only a single intensity value({min_val})')  
    return np.array(new_stack)  
    

def find_neighbors(mask, quantification_table): 
    print('Calculating neighbors - this might take some time')  
    # Dictionary to store cell contact information
    neighbor_id = {}
    # Iterate through each segmented frame
    for frame_idx, frame in enumerate(mask):
        frame_neighbor_id = {}
        unique_cells = np.unique(frame)
        # Remove background label (assuming it's 0)
        unique_cells = unique_cells[unique_cells != 0]
        # Extract contours for each cell
        contours = {}
        for particle_id in unique_cells:
            # Create binary mask for the current cell
            binary_mask = (frame == particle_id).astype(np.uint8)
            # Find contours
            cont, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cont:  # Add contour if found
                contours[particle_id] = cont[0]  # Only the first contour is relevany
        # Check for cell contacts
        for particle_id1, contour1 in contours.items():
            frame_neighbor_id[particle_id1] = set()
            for particle_id2, contour2 in contours.items():
                if particle_id1 == particle_id2:
                    continue  # Avoid self-comparison or redundant checks     
                # Calculate distances between contour points
                dist = cdist(contour1[:, 0, :], contour2[:, 0, :])  # (x, y) points
                if np.min(dist) <= 2:  # Threshold for contact (# pixel)
                    frame_neighbor_id[particle_id1].add(particle_id2)
                    if particle_id2 not in frame_neighbor_id:
                        frame_neighbor_id[particle_id2] = set()
                    frame_neighbor_id[particle_id2].add(particle_id1)
        # Add frame contacts to the main dictionary
        neighbor_id[frame_idx] = frame_neighbor_id
    ## adding the neighbor_id to the quantification table
    quantification_table['neighbor_id'] = None
    for frame in neighbor_id :
        for particle in neighbor_id[frame]: 
                quantification_table.at[
                quantification_table[(quantification_table['frame'] == frame) & 
                                    (quantification_table['particle'] == particle)].index[0], 'neighbor_id'] = sorted(neighbor_id[frame][particle])

    ## return the result
    return quantification_table
    

def create_centroid_circles_stack(mask, quantification_table, radius = 5):
    quantification_groupby_frame = quantification_table.groupby('frame')
    centroid_circles_stack = []
    for i, frame in enumerate(mask): 
        quantification = quantification_groupby_frame.get_group(i) 
        new_frame = np.zeros_like(frame, dtype = np.uint16)
        # add the centroid as a circle    
        for _, row in quantification.iterrows():
            centroid_y, centroid_x = int(row['centroid_y']), int(row['centroid_x'])
            particle_id = int(row['particle'])
            # generate coordinates for a circle centered at the centroid
            rr, cc = disk((centroid_y, centroid_x), radius, shape=frame.shape)
            # assign the particle id as the intensity for the circle area
            new_frame[rr, cc] = particle_id
        # append to my list       
        centroid_circles_stack.append(new_frame)
    centroid_circles_stack = np.array(centroid_circles_stack, dtype=np.uint16)

    ## return the result
    return centroid_circles_stack.astype("int32")


def create_centroid_particleIDs_stack(mask, quantification_table, font_size=12):
    quantification_groupby_frame = quantification_table.groupby('frame')
    centroid_particleIDs_stack = []
    font = ImageFont.load_default()
    for i, frame in enumerate(mask):
        quantification = quantification_groupby_frame.get_group(i)
        # Create empty image (single intensity for all labels, e.g., 1)
        img = Image.new('I;16', (frame.shape[1], frame.shape[0]), 0)
        draw = ImageDraw.Draw(img)
        for _, row in quantification.iterrows():
            centroid_y, centroid_x = int(row['centroid_y']), int(row['centroid_x'])
            # Use the same intensity (e.g., 1) for all labels
            draw.text(
                (centroid_x, centroid_y),
                str(int(row['particle'])),
                fill=1,  # constant intensity for all labels
                font=font,
                anchor='mm')
        new_frame = np.array(img, dtype=np.uint16)
        centroid_particleIDs_stack.append(new_frame)
    centroid_particleIDs_stack = np.array(centroid_particleIDs_stack, dtype=np.uint16)
    ## return the result
    return centroid_particleIDs_stack.astype("int32")


def get_neighbor_annotation(quantification_table): 
    """Get info about neighbor annotation and calculate some metrics"""

    ## looping over data to create dictionary
    neighbor_id  = {}
    quant_by_frame = quantification_table.groupby('frame')
    for frame, table in quant_by_frame: 
        neighbor_id[frame] = dict(zip(table['particle'], table['neighbor_id']))
    
    ## getting the annotation info for the neighbors and storing in another dict
    neighbor_anno_dict = {}
    for frame in neighbor_id :
        neighbor_anno_dict[frame] = {}
        for particle in neighbor_id[frame]: 
            particle_neighbor_id = neighbor_id[frame][particle]
            particle_neighbor_anno = []
            for ID in particle_neighbor_id: 
                particle_neighbor_anno.append(quantification_table.loc[quantification_table['particle'] == int(ID), 'annotation'].values[0])
            neighbor_anno_dict[frame][particle] = particle_neighbor_anno

    ## adding the neighbors annotation to the quantification table
    quantification_table['neighbor_anno'] = None
    for frame in neighbor_anno_dict:
        for particle in neighbor_anno_dict[frame]: 
            quantification_table.at[
                quantification_table[(quantification_table['frame'] == frame) & 
                                     (quantification_table['particle'] == particle)].index[0], 'neighbor_anno'] = neighbor_anno_dict[frame][particle]    
    
        ## calculating some additional basic metrics
    quantification_table['n_neighbors'] = quantification_table['neighbor_anno'].apply(lambda x: len(x))
    quantification_table['n_neighbors_other'] = quantification_table['neighbor_anno'].apply(lambda x: x.count('other'))
    quantification_table['n_neighbors_de'] = quantification_table['neighbor_anno'].apply(lambda x: x.count('DE'))
    quantification_table['n_neighbors_ve'] = quantification_table['neighbor_anno'].apply(lambda x: x.count('VE'))
    quantification_table['n_neighbors_exve'] = quantification_table['neighbor_anno'].apply(lambda x: x.count('exVE'))
    # returning the dataframe adn file_path
    return quantification_table

def neighbor_change_quantification(quantification_table): 
    # Add new columns
    quantification_table['neighbor_change_from_initial'] = None
    quantification_table['n_neighbor_change_from_initial'] = np.nan
    quantification_table['neighbor_change_from_previous'] = None
    quantification_table['n_neighbor_change_from_previous'] = np.nan
    # Sort the dataframe
    quantification_table = quantification_table.sort_values(['particle', 'frame']).reset_index(drop=True)
    
    for particle, table in quantification_table.groupby('particle'):
        initial_neighbors = set(table.iloc[0]['neighbor_id'])
        
        prev_neighbors = None
        neighbor_change_from_initial = []
        n_neighbor_change_from_initial = []
        neighbor_change_from_previous = []
        n_neighbor_change_from_previous = []
        
        for neighbors in table['neighbor_id']:
            current = set(neighbors)
        
            # Change from frame 0
            change_initial = current.symmetric_difference(initial_neighbors) if current.symmetric_difference(initial_neighbors) else None
            change_initial = change_initial if change_initial else None
            n_change_initial = int(len(change_initial)) if change_initial else int(0)
            neighbor_change_from_initial.append(change_initial)
            n_neighbor_change_from_initial.append(n_change_initial)
        
            # Change from previous frame
            if prev_neighbors is None:
                neighbor_change_from_previous.append(None)
                n_neighbor_change_from_previous.append(np.nan)
            else:
                change_previous = current.symmetric_difference(prev_neighbors)
                change_previous = change_previous if change_previous else None
                n_change_previous = int(len(change_previous)) if change_previous else int(0)
                neighbor_change_from_previous.append(change_previous)
                n_neighbor_change_from_previous.append(n_change_previous)
            
            prev_neighbors = current
            
        quantification_table.loc[quantification_table['particle'] == particle, 'neighbor_change_from_initial'] = neighbor_change_from_initial
        quantification_table.loc[quantification_table['particle'] == particle, 'n_neighbor_change_from_initial'] = n_neighbor_change_from_initial
        quantification_table.loc[quantification_table['particle'] == particle, 'neighbor_change_from_previous'] = neighbor_change_from_previous
        quantification_table.loc[quantification_table['particle'] == particle, 'n_neighbor_change_from_previous'] = n_neighbor_change_from_previous
    return quantification_table


def more_metrics_and_summary(quantification_table):
    """
    Compute displacement, velocity, acceleration, turning angles, and other statistics 
    for particle trajectories linked by TrackPy.

    Parameters:
    - quantification_table: DataFrame with linked particle trajectories containing 'x', 'y', 'frame', 'particle'.

    Returns:
    - quantification_table: Updated DataFrame with calculated statistics.
    - summary_table: Summary statistics aggregated per particle.
    """
    # Step 1: Sort the data by particle and frame
    quantification_table = quantification_table.sort_values(by=['particle', 'frame'])

    # Step 2: Compute displacement, velocity, and acceleration
    quantification_table['dx'] = quantification_table.groupby('particle')['centroid_x'].diff()
    quantification_table['dy'] = quantification_table.groupby('particle')['centroid_y'].diff()
    quantification_table['displacement'] = np.sqrt(quantification_table['dx']**2 + quantification_table['dy']**2)
    quantification_table['velocity'] = quantification_table['displacement'] 
    quantification_table['acceleration'] = quantification_table.groupby('particle')['velocity'].diff() 


    # Step 3: Normalize displacement vectors
    quantification_table['dx_norm'] = quantification_table['dx'] / quantification_table['displacement']
    quantification_table['dy_norm'] = quantification_table['dy'] / quantification_table['displacement']

    # Step 4: Compute turning angles
    quantification_table['dot_product'] = (
        quantification_table.groupby('particle')['dx_norm'].shift(1) * quantification_table['dx_norm'] +
        quantification_table.groupby('particle')['dy_norm'].shift(1) * quantification_table['dy_norm']
    )
    quantification_table['dot_product'] = quantification_table['dot_product'].clip(-1, 1)  # Avoid numerical errors
    quantification_table['turning_angle'] = np.arccos(quantification_table['dot_product']) ## in radians

    # Step 6: Aggregate statistics for each particle
    summary_table = quantification_table.groupby('particle').agg(
        mean_velocity=('velocity', 'mean'),
        mean_acceleration=('acceleration', 'mean'),
        mean_turning_angle=('turning_angle', 'mean'),
        total_displacement=('displacement', 'sum'),
        track_length=('frame', 'count'), 
        mean_area = ('area', 'mean'), 
        mean_perimeter = ('perimeter', 'mean'), 
        mean_eccentricity = ('eccentricity', 'mean'), 
        mean_n_neighbors = ('n_neighbors', 'mean'), 
        mean_n_neighbor_change_from_previous = ('n_neighbor_change_from_previous', 'mean'), 
        mean_n_neighbor_change_from_initial = ('n_neighbor_change_from_initial', 'mean'), 
        x_inital = ('centroid_x', 'first'), 
        x_final = ('centroid_x', 'last'), 
        y_initial = ('centroid_y', 'first'), 
        y_final = ('centroid_y', 'last'))
    summary_table['particle'] = summary_table.index.values.copy()

    ## calculating net displacement
    x_delta = summary_table['x_inital'] - summary_table['x_final']
    y_delta = summary_table['y_initial'] - summary_table['y_final']
    net_displacement = np.sqrt(x_delta**2 + y_delta**2)
    summary_table['net_displacement_normalized'] = net_displacement * len(quantification_table['frame'].unique()) / summary_table['track_length']

    ## getting the annotation values
    annotation_dictionary  = dict(zip(quantification_table['particle'], quantification_table['annotation']))
    summary_table['annotation'] = summary_table['particle'].map(annotation_dictionary)
    # Normalize total displacement based on image stack shape (optional)
    summary_table['normalized_displacement'] = (summary_table['total_displacement'] * len(quantification_table['frame'].unique()) / summary_table['track_length'])
    
    return quantification_table, summary_table

def create_custom_table(quantification_table, include_columns, query_by, melt_by, group_by): 
    custom_table = quantification_table.copy()
    custom_table = custom_table[include_columns].copy()
    if query_by:
        print(f"Query: {query_by}")
        try: 
            custom_table = custom_table.query(query_by)
        except Exception as e: 
            print(f"Invalid query expression: {e}")
    if melt_by[0] is not None: 
        custom_table = custom_table.melt(id_vars = melt_by)
    if group_by[0] is not None:
        custom_table = custom_table.groupby(group_by[0])
    return custom_table


### code for edges and vertices -------------------------------

def define_background(mask, bckgrdlbl = int(999)):
    mask = mask.copy()
    background = (mask == 0)
    mask_background, num_features = label(background)
    label_areas = np.bincount(mask_background.ravel())
    label_areas[0] = 0  # 0 (background) in this case is going to be the embryo - we are not interested in that
    true_background_label = np.argmax(label_areas)
    mask_background = (mask_background == true_background_label)
    # set the background mask to the backgroundlbl
    mask[mask_background] = bckgrdlbl
    return mask, bckgrdlbl

def create_mask_from_dict(mask, dict_data):
    """
    Create masks for edges and vertices from the results dictionary.

    Parameters:
        mask_shape (tuple): Shape of the original mask (T, H, W)
        dict_data (dict): Output of analyze_edges_and_vertices()

    Returns:
        mask_of_interest: Two np.arrays with IDs as pixel values
    """
    mask_shape = mask.shape
    T, H, W = mask_shape
    mask_of_interest = np.zeros((T, H, W), dtype=np.int32)

    for t in range(T):
        for j_id, j_data in dict_data[t].items():
            for y, x in j_data["coords"]:
                mask_of_interest[t, y, x] = j_data["id"]

    return mask_of_interest

def get_pixel_neighborhood(y, x, frame, padding = 1):
    if len(frame.shape) == 3:
        raise ShapeError("A single frame has to be passed (2D array expected).")
  
    ## defining min and max to not run into any issues at the borders
    y_min = 0
    y_max = frame.shape[0]
    x_min = 0
    x_max = frame.shape[1]

    # defining start and stop of kernel (keep in mind the stop index itself will not be part of the kernel)
    y_start = max(y_min, y - padding)
    y_stop = min(y_max, y + padding + 1) ## increasing by one because index will be used as a stop (not part of the kernel)
    x_start = max(x_min, x - padding)
    x_stop = min(x_max, x + padding + 1)

    pixel_neighborhood = frame[y_start: y_stop, x_start:x_stop] 

    return pixel_neighborhood

def dilate_labels(mask, kernel_dimension=5):
    if kernel_dimension == 0: 
        return mask
    elif kernel_dimension % 2 == 0 : 
        raise ValueError("The kernel dimensions have to be odd for a symmetric dilation")
    
    else:
        footprint = np.ones((kernel_dimension, kernel_dimension))
        # Use grey_dilation to "push out" the labels — fast!
        output = np.zeros_like(mask)
        for i, frame in enumerate(mask):
            output[i] = grey_dilation(frame, footprint=footprint)
        return output


def relabel_vertices_mask(mask):
    output = np.zeros(mask.shape)

    for i, frame in enumerate(mask): 
        output[i], n_features = label(frame)

    

    return output.astype('int32')

## for the edges we immediately create the dictionary, because there Id's will not change


def get_edges_dictionary(mask, outline, padding = 2, min_area = 5):
    """
    Analyze cell edges across timepoints.

    Parameters:
        mask (ndarray): 3D array of shape (T, H, W) with cell IDs.
        outline (ndarray): 3D array of shape (T, H, W) with outlines (0 or 1).
    
    Returns:
        dict: Dictionary with edges for each timepoint.
    """
    T, H, W = mask.shape
    results = {}
    
    id_counter = 0

    for t in range(T):
        current_edge = {}
        edge_map = {}

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if outline[t, y, x] == 0:
                    continue

                neighborhood = get_pixel_neighborhood(y,x, mask[t], padding = padding).flatten()
                neighbor_ids = set(neighborhood)
                neighbor_ids.discard(0)

                if len(neighbor_ids) == 2:
                    neighbor_ids = tuple(sorted(neighbor_ids))
                    if neighbor_ids not in edge_map:
                        edge_map[neighbor_ids] = []
                    edge_map[neighbor_ids].append((y, x))

        # Assign IDs and calculate properties
        for neighbor_ids, coords in edge_map.items():
            coords = np.array(coords)
            if len(coords) < min_area: 
                continue
            centroid = coords.mean(axis=0)
            id_str = int(id_counter)
            current_edge[id_str] = {
                "cell_ids": list(neighbor_ids),
                "coords": coords,
                "centroid": tuple(centroid), 
                "id": int(id_counter)
            }
            id_counter += 1

        results[t] = current_edge

    return results


## here I first run the algorithm to identify vertices, dilate them, label (merge) them, and then create the dict

def get_intitial_vertices_dictionary(mask, outline, padding = 1, min_area = 1):
    """
    Analyze cell edges across timepoints.

    Parameters:
        mask (ndarray): 3D array of shape (T, H, W) with cell IDs.
        outline (ndarray): 3D array of shape (T, H, W) with outlines (0 or 1).
    
    Returns:
        dict: Dictionary with edges for each timepoint.
    """
    T, H, W = mask.shape
    results = {}
    
    id_counter = 0

    for t in range(T):
        current_vertex = {}
        vertex_map = {}

        for y in range(1, H - 1):
            for x in range(1, W - 1):
                if outline[t, y, x] == 0:
                    continue


                neighborhood = get_pixel_neighborhood(y,x, mask[t], padding = padding).flatten()
                neighbor_ids = set(neighborhood)
                neighbor_ids.discard(0)

                if len(neighbor_ids) > 2:
                    neighbor_ids = tuple(sorted(neighbor_ids))
                    if neighbor_ids not in vertex_map:
                        vertex_map[neighbor_ids] = []
                    vertex_map[neighbor_ids].append((y, x))

        # Assign IDs and calculate properties
        for neighbor_ids, coords in vertex_map.items():
            coords = np.array(coords)
            if len(coords) < min_area: 
                continue
            id_str = int(id_counter)
            current_vertex[id_str] = {
                "coords": coords,
                "id": int(id_counter)
            }
            id_counter += 1

        results[t] = current_vertex

    return results


def get_vertices_id_dict(vertices_mask, mask_bckgrdlbl):
    if vertices_mask.shape != mask_bckgrdlbl.shape:
        raise InputError("The dimensions of the vertices mask and the background mask are not identical")

    vertices_dict = {}
    T, Y, X = vertices_mask.shape

    for t in range(T): 
        frame_vertices = vertices_mask[t]
        frame_bckgrdlbl = mask_bckgrdlbl[t] 
        ids_vertices = set(np.unique(frame_vertices))
        ids_vertices.discard(0)

        frame_dict = {}
        for id_vertex in ids_vertices: 
            id_neighbors = set(np.unique(frame_bckgrdlbl[frame_vertices == id_vertex]))
            id_neighbors.discard(0)
            frame_dict[id_vertex] = list(id_neighbors)

        vertices_dict[t] = frame_dict
    return vertices_dict

        
def edges_vertices_post_basic_quantification(dictionary, quantification): 
    quantification['neighbor_ids'] = quantification.apply(lambda row: dictionary.get(row['frame'], {}).get(row['initial_id'], []),axis=1) 
    quantification['degree'] = quantification['neighbor_ids'].apply(len)
    quantification['boundary'] = quantification['neighbor_ids'].apply(lambda neighbors: 999 in neighbors)
    
    return quantification

def get_edges_id_dict(dictionary):
    return {
        frame: {
            edge_id: edge_data["cell_ids"]
            for edge_id, edge_data in edges.items()
        }
        for frame, edges in dictionary.items()
    }


#### EDGES AND VERTICES CELL NEIGHBOR CALCULATIONS


def get_dictionary_keyEdgeVertexId_valueCellId(mask_edges_vertices, mask_bckgrdlbl): 
    if mask_edges_vertices.shape != mask_bckgrdlbl.shape: 
        raise DimensionError("The dimensions of the edges/vertices mask and the cell mask are not identical")
    
    T, Y, X = mask_edges_vertices.shape
    nested_frame_id_dictionary = {}

    
    for t in range(T): 
        nested_frame_id_dictionary[t] = {}
        edges_vertices_ids = np.unique(mask_edges_vertices[t])
        edges_vertices_ids = edges_vertices_ids[edges_vertices_ids != 0]
        for edge_vertex_id in edges_vertices_ids: 
            neighbors = []
            mask_id = (mask_edges_vertices[t] == edge_vertex_id)
            y_id, x_id = np.where(mask_id)
            for y, x in zip(y_id, x_id):
                neighbor = get_pixel_neighborhood(y, x, mask_bckgrdlbl[t], padding = 1).flatten()
                neighbors.extend(neighbor)
            ## adding values to the dictionary
            neighbors = np.array(neighbors)
            neighbors = neighbors[neighbors != 0]
            neighbors = np.unique(neighbors)
            nested_frame_id_dictionary[t][edge_vertex_id] = neighbors.tolist()
    return nested_frame_id_dictionary


def map_cellId_to_cellAnno(quantification_edges_vertices, quantification_cells):
    # Build a mapping from particle id to annotation once
    cell_to_anno = quantification_cells.set_index('particle')['annotation'].to_dict()
    cell_to_anno[999] = 'background'
    # Apply mapping
    quantification_edges_vertices['cell_anno'] = quantification_edges_vertices['cell_ids'].apply(
        lambda neighbor_anno: [cell_to_anno[neighbor_id] for neighbor_id in neighbor_anno if neighbor_id in cell_to_anno])
    return quantification_edges_vertices

def map_cellId_to_cellAnno(quantification_edges_vertices, quantification_cells):
    # Build a mapping from particle id to annotation once
    cell_to_anno = quantification_cells.set_index('particle')['annotation'].to_dict()
    cell_to_anno[999] = 'background'
    # Apply mapping
    quantification_edges_vertices['cell_anno'] = quantification_edges_vertices['cell_ids'].apply(
        lambda neighbor_anno: [cell_to_anno[neighbor_id] for neighbor_id in neighbor_anno if neighbor_id in cell_to_anno])
    return quantification_edges_vertices

def map_nested_dictionary_to_ids(quantification_table, nested_frame_id_dictionary, column_name, identifiers = 'particle'): 
    quantification_table[column_name] = None
    
    for frame in nested_frame_id_dictionary:
        for identifier in nested_frame_id_dictionary[frame]:
            mask = (quantification_table['frame'] == frame) & (quantification_table[identifiers] == identifier)
            matching_rows = quantification_table[mask]

            if not matching_rows.empty:
                value = nested_frame_id_dictionary[frame].get(identifier, [])

                if isinstance(value, list):
                    quantification_table.at[matching_rows.index[0], column_name] = sorted(value)
                elif isinstance(value, (bool, np.bool_)):
                    quantification_table.at[matching_rows.index[0], column_name] = value
                else:
                    quantification_table.at[matching_rows.index[0], column_name] = value

    return quantification_table

def get_dictionary_keyCellId_valueEdgeVertexId(quantification_edges_vertices, identifiers = 'particle'):
    # make nested dictionary
    # first key is the frame DONE
    # second key is:
            # create a unique list with all the ids in quantification_edges_cell[cell_ids] DONE
            # create a mask that is true for each row of quantification edges where unique_id is present in quantification_edges_cell[cell_ids]
            # use this mask to extract quantification_edges[particl]

    cell_id_to_edge_vertex_id_dictionary = {}

    for frame in np.unique(quantification_edges_vertices['frame']):
        cell_id_to_edge_vertex_id_dictionary[frame] = {}
        frame_data = quantification_edges_vertices[quantification_edges_vertices['frame'] == frame]
        unique_cell_ids = [cell_id for cell_ids in frame_data['cell_ids'] for cell_id in cell_ids]
        unique_cell_ids = np.unique(unique_cell_ids)
        for cell_id in unique_cell_ids:
            cell_id_to_edge_vertex_id_dictionary[frame][cell_id] = []
            mask = [cell_id in cell_ids for cell_ids in  frame_data['cell_ids']]
            cell_id_to_edge_vertex_id_dictionary[frame][cell_id].extend(frame_data.loc[mask, identifiers])
        
    return cell_id_to_edge_vertex_id_dictionary


def get_cell_boundary_annotation(quantification_edges):
    boundary_cell_dictionary = {}

    for frame in np.unique(quantification_edges['frame']):
        boundary_cell_dictionary[frame] = {}
        frame_data = quantification_edges[quantification_edges['frame'] == frame]
        unique_cell_ids = [cell_id for cell_ids in frame_data['cell_ids'] for cell_id in cell_ids]
        unique_cell_ids = np.unique(unique_cell_ids)
        for cell_id in unique_cell_ids:
            boundary_cell_dictionary[frame][cell_id] = []
            mask = [cell_id in cell_ids for cell_ids in  frame_data['cell_ids']]
            boundary_cell_dictionary[frame][cell_id] = np.any(frame_data.loc[mask, 'boundary'])

    
    return boundary_cell_dictionary


def get_signal_quantification(image, mask, quantification_table):
    T, Y, X = mask.shape
    sum_dictionary = {}
    mean_dictionary = {}
    for t in range(T): 
        sum_dictionary[t] = {}
        mean_dictionary[t] = {}
        mask_t = mask[t] 
        image_t = image[t]
        identifiers = np.unique(mask_t)
        identifiers = identifiers[identifiers != 0]
        for identifier in identifiers: 
            identifier_localization = mask_t == identifier
            signal_sum = np.sum(image_t[identifier_localization])
            signal_mean = np.mean(image_t[identifier_localization])
            sum_dictionary[t][identifier] = signal_sum
            mean_dictionary[t][identifier] = signal_mean
    ## mapping the dictionary to the table
    quantification_table = map_nested_dictionary_to_ids(quantification_table, sum_dictionary, column_name = 'signal_sum')
    quantification_table = map_nested_dictionary_to_ids(quantification_table, mean_dictionary, column_name = 'signal_mean')

    return quantification_table


def track_edges_vertices_alternative_approach(mask, quantification_table, neighbor_configs, t_column = 'frame', tracking_id = 'particle'):
    print('Tracking the features over time and relabeling the mask')

    quantification_table[tracking_id] = None
    tracking_counter = 1
    # Normalize neighbor config lists: sort and convert to tuples
    quantification_table["_neighbor_config_tuple"] = quantification_table[neighbor_configs].apply(lambda x: tuple(sorted(x)))
    
    # Get unique neighbor config tuples
    unique_neighbor_configs = quantification_table["_neighbor_config_tuple"].unique()
    
    for unique_neighbor_config in unique_neighbor_configs:
        config_mask = quantification_table["_neighbor_config_tuple"] == unique_neighbor_config
        quantification_table.loc[config_mask, tracking_id] = tracking_counter
        tracking_counter += 1
    
    # Optionally drop the helper column
    quantification_table.drop(columns="_neighbor_config_tuple", inplace=True)

    mask_relabeled = []
    # Group tables by frame
    grouped_tables = quantification_table.groupby('frame')
    # Iterate over each frame and in mask
    for t, frame in enumerate(mask):
        new_frame = np.zeros_like(frame, dtype=np.int32)  # Initialize a new frame with zeros
        if t in grouped_tables.groups:
            # Get the tables for this frame
            frame_props = grouped_tables.get_group(t)
            # Iterate over each row in tables for this frame
            for _, row in frame_props.iterrows():
                segmentation_id = row['initial_id']
                particle_id = row['particle']
                # Replace regions with the particle number
                new_frame[frame == segmentation_id] = particle_id
        # Append the updated frame to the new stack
        mask_relabeled.append(new_frame)

    mask_relabeled = np.array(mask_relabeled)
    ## small test----------------------------------------
    print(f'New mask shape: {mask_relabeled.shape}, type: {type(mask_relabeled)}, dtype {mask_relabeled.dtype}')
    print(quantification_table.head()) 

    ## return the results
    return quantification_table, mask_relabeled


