"""
My first napari plugin for cell movement quantification
"""

from typing import TYPE_CHECKING
from napari.types import LabelsData
from magicgui import magic_factory, magicgui
import napari
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import os
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog
from magicgui.widgets import Select, PushButton, Container
from magicgui.widgets import ComboBox, LineEdit

from ._functions import *

global_plugin_data = {'annotate_cells_initiation' : True}

viewer = "napari.Viewer"

@magic_factory(call_button="Load table, label, or image", data_type = {"choices": ['label', 'image', 'table']})
def load_data_widget(viewer: "napari.Viewer", 
              data_type: str = 'table'):
    load_data(viewer, global_plugin_data, data_type)

@magic_factory(call_button="Track and quantify")
def track_and_quantify(viewer: "napari.Viewer",
                    mask_layer: "napari.layers.Labels",
                    search_range: int = 20, 
                    memory: int = 4, 
                    centroid_radius: float = 5, 
                    centroid_font_size = 5):
    mask = mask_layer.data  # extracting numpy data from the selected napari layer
    # calling functions for centroids, tracking, quatnification and neighbors
    quantification_cells = centroids_and_basic_quantification(mask)
    quantification_cells, mask = link_centroids_relabel_mask(mask, quantification_cells, search_range=search_range, memory=memory)
    ## Saving the relabeled mask
    file_path = save_data(viewer, global_plugin_data, mask, data_type = 'label', file_path_default = "mask_cells.tif", message = "Save the relabeled mask to disk")    
    if file_path is not None:
        viewer.layers.remove(mask_layer)  # remove the original layer directly
        viewer.add_labels(mask, name=os.path.basename(file_path)) ## update the viewer with the new mask
    # finding neighbors and saving
    quantification_cells = find_neighbors(mask,quantification_cells)
    save_data(viewer, global_plugin_data, quantification_cells, data_type = 'table', file_path_default = "quantification_cells.tsv", message = "Save the quantification table to disk")    
    #centroids circle and saving
    #mask_cells_centroids = create_centroid_circles_stack(mask, quantification_cells, radius = centroid_radius)
    #file_path = save_data(viewer, global_plugin_data, mask_cells_centroids, data_type = 'label', file_path_default = "mask_cells_centroids.tif", message = "Save the mask with centroids as circles to disk")    
    #if file_path is not None:
    #    viewer.add_labels(mask_cells_centroids, name=os.path.basename(file_path)) ## update the viewer with the new mask
    # centroids id and saving
    mask_cells_ids = create_centroid_particleIDs_stack(mask, quantification_cells, font_size = centroid_font_size)
    file_path = save_data(viewer, global_plugin_data, mask_cells_ids, data_type = 'label', file_path_default = "mask_cells_ids.tif", message = "Save the mask with centroids as particle ID to disk")    
    if file_path is not None:
        viewer.add_labels(mask_cells_ids, name=os.path.basename(file_path)) ## update the viewer with the new mask
    
    print("Tracking and qantification successfully finished!")



@magic_factory(call_button="Start Annotation / Switch Annotation Value", annotation={"choices": ["DE", "VE", "exVE", "other"]})
def annotate_cells(viewer: "napari.Viewer",
                   mask_layer: "napari.layers.Labels",
                   annotation: str = "DE"):

    ## RESET POINT
    
    if global_plugin_data['annotate_cells_initiation']:
        ## create all the new masks
        global_plugin_data["other_layer"] = viewer.add_labels(mask_layer.data.copy(), name = 'other')
        global_plugin_data["DE_layer"] = viewer.add_labels(np.zeros_like(mask_layer.data), name = 'definitive_endoderm')
        global_plugin_data["VE_layer"] = viewer.add_labels(np.zeros_like(mask_layer.data), name = 'visceral_endoderm')
        global_plugin_data["exVE_layer"] = viewer.add_labels(np.zeros_like(mask_layer.data), name = 'exVE')
        global_plugin_data['annotate_cells_initiation'] = False

    ## set the other_layer as active
    viewer.layers.selection.active = global_plugin_data["other_layer"]

    # Persistent state stored in the layer metadata (avoids reloading)
    if 'quant_table' not in global_plugin_data["other_layer"].metadata:
        global_plugin_data["other_layer"].metadata['quant_table'],  global_plugin_data["other_layer"].metadata['file_path'] = load_data(viewer, global_plugin_data, data_type = 'table', message = 'Load the quantification table that you want to annotate')

    # Update current annotation state
    global_plugin_data["other_layer"].metadata['current_annotation'] = annotation
    print(f"Annotation mode set to '{annotation}'.")

    # Define the click callback if not already defined
    if 'callback_connected' not in global_plugin_data["other_layer"].metadata:

        def click_callback(layer, event):
            coords = tuple(np.round(event.position).astype(int))

            if len(coords) == 3:
                z, y, x = coords

            else:
                print("Unsupported dimensionality.")
                return

            particle_id = layer.data[z, y, x]

            if particle_id == 0:
                print("Clicked background, ignoring.")
                return

            current_annotation = layer.metadata['current_annotation']
            quant_table = layer.metadata['quant_table']

            # Update annotation in the table
            quant_table.loc[(quant_table['particle'] == particle_id), 'annotation'] = current_annotation

            ## update the masks
            index_mask = global_plugin_data['other_layer'].data == particle_id
            global_plugin_data['other_layer'].data[index_mask] = 0
            global_plugin_data[f'{current_annotation}_layer'].data[index_mask] = particle_id

            print(f"Annotated particle {particle_id} as '{current_annotation}'.")

            # Refresh display
            layer.refresh()

        # Clear previous callbacks to ensure only one is active
        global_plugin_data["other_layer"].mouse_drag_callbacks.clear()
        global_plugin_data["other_layer"].mouse_drag_callbacks.append(click_callback)
        global_plugin_data["other_layer"].metadata['callback_connected'] = True
        print("Callback connected. Start clicking cells to annotate.")

def annotate_cells_with_save_button():
    widget = annotate_cells()

    # Create a custom QWidget container
    container_widget = QWidget()
    layout = QVBoxLayout()

    # Add magicgui-generated widget directly
    layout.addWidget(widget.native)

    # Create and configure the save button
    save_button = QPushButton("Save Quantification Table")

    def on_save_button_clicked():
        quant_table = global_plugin_data["other_layer"].metadata.get('quant_table', None)
        file_path = global_plugin_data["other_layer"].metadata.get('file_path', '')
        viewer = widget.viewer.value
        if (quant_table is None) or (file_path is None):
            print("global_plugin_data['other_layer'] is either none or missing quant_table or file_path")
            return

        save_data(
            viewer, 
            global_plugin_data,
            data=quant_table, 
            data_type='table', 
            file_path_default=file_path, 
            message='Saving quantification table with updated annotation')
        
        other_layer_data = global_plugin_data['other_layer'].data
        save_data(
            viewer, 
            global_plugin_data,
            data=other_layer_data,
            file_path_default = os.path.join(os.path.dirname(file_path), 'other.tif'),
            data_type='label',
            message='Saving lables of "other" cells')
        
        de_layer_data = global_plugin_data['DE_layer'].data
        save_data(
            viewer, 
            global_plugin_data,
            data=de_layer_data,
            file_path_default = os.path.join(os.path.dirname(file_path), 'definitive_endoderm.tif'),
            data_type='label',
            message='Saving lables of definitive endoderm cells')
        
        ve_layer_data = global_plugin_data['VE_layer'].data
        save_data(
            viewer, 
            global_plugin_data,
            data=ve_layer_data,
            file_path_default = os.path.join(os.path.dirname(file_path), 'visceral_endoderm.tif'),
            data_type='label',
            message='Saving lables of definitive endoderm cells')
        
        exve_layer_data = global_plugin_data['exVE_layer'].data
        save_data(
            viewer, 
            global_plugin_data,
            data=exve_layer_data,
            file_path_default = os.path.join(os.path.dirname(file_path), 'exVE.tif'),
            data_type='label',
            message='Saving lables of definitive endoderm cells')

    save_button.clicked.connect(on_save_button_clicked)
    # Add save button below the widget
    layout.addWidget(save_button)

    container_widget.setLayout(layout)
    return container_widget

@magic_factory(call_button="Start Post Annotation Quantification")
def post_annotation_quantification(viewer: "napari.Viewer"): 
    quantification_cells, file_path = load_data(viewer, global_plugin_data, data_type = 'table', message = 'Load the quantification table')
    quantification_cells = get_neighbor_annotation(quantification_cells)
    quantification_cells = neighbor_change_quantification(quantification_cells)
    quantification_cells, summary_cells = more_metrics_and_summary(quantification_cells)
    save_data(viewer, global_plugin_data, quantification_cells, data_type = 'table', file_path_default = file_path, message = "Save the quantification table to disk")
    save_data(viewer, global_plugin_data, summary_cells, data_type = 'table', file_path_default = 'summary_cells.tsv', message = "Save the summary table to disk")
    print('Post annotation quantification successfully finished')

@magic_factory(call_button="Quantify edges and vertices")
def run_edges_vertices_pipeline(viewer: "napari.Viewer",
                                image_layer: "napari.layers.Image",
                                mask_layer: "napari.layers.Labels", 
                                outline_layer: "napari.layers.Labels", 
                                edge_discovery_padding: int = 2, 
                                edge_min_area: int = 8,
                                edge_dilation_kernel: int = 0, 
                                vertex_discovery_padding: int = 1, 
                                vertex_min_area: int = 1, 
                                vertex_dilation_kernel = 5, 
                                centroid_font_size = 3):  
    
    quantification_cells, file_path_quantification_cells = load_data(viewer, global_plugin_data, data_type = 'table', message = 'Load the cell quantification table')

    print("Starting edge and vertex assignment and quantification, that might take a while")
    image = image_layer.data
    mask = mask_layer.data
    outline = outline_layer.data
    ## RUN FUNCTIONS IN CORRECT SEQEUNCE 
    mask_bckgrdlbl, bckgrdlbl =define_background(mask)
    
    edges_dict =get_edges_dictionary(mask_bckgrdlbl, outline, padding = edge_discovery_padding, min_area = edge_min_area)
    mask_edges =create_mask_from_dict(mask_bckgrdlbl, edges_dict)
    mask_edges =dilate_labels(mask_edges, kernel_dimension = edge_dilation_kernel)
    edges_id_dict =get_edges_id_dict(edges_dict)

    
    vertices_initial_dict =get_intitial_vertices_dictionary(mask_bckgrdlbl, outline, padding = vertex_discovery_padding, min_area = vertex_min_area)
    mask_vertices =create_mask_from_dict(mask_bckgrdlbl, vertices_initial_dict)
    mask_vertices =dilate_labels(mask_vertices, kernel_dimension = vertex_dilation_kernel)
    mask_vertices =relabel_vertices_mask(mask_vertices)
    vertices_id_dict =get_vertices_id_dict(mask_vertices, mask_bckgrdlbl)
    
    quantification_edges =centroids_and_basic_quantification(mask_edges)
    quantification_vertices =centroids_and_basic_quantification(mask_vertices)
    quantification_edges =edges_vertices_post_basic_quantification(edges_id_dict, quantification_edges)
    quantification_vertices =edges_vertices_post_basic_quantification(vertices_id_dict, quantification_vertices)

    ## PUT ALL THE CELL IDS, ANNOTATIONS AND EDGE IDS, VERTEX IDS TOGETHER
    ## edges stuff
    keyEdgeId_valueCellId = get_dictionary_keyEdgeVertexId_valueCellId(mask_edges, mask_bckgrdlbl)
    quantification_edges = map_nested_dictionary_to_ids(quantification_edges, keyEdgeId_valueCellId, column_name = 'cell_ids',  identifiers = 'initial_id')
    quantification_edges = map_cellId_to_cellAnno(quantification_edges, quantification_cells)
    keyCellId_valueEdgeId = get_dictionary_keyCellId_valueEdgeVertexId(quantification_edges, identifiers = 'initial_id')
    quantification_cells = map_nested_dictionary_to_ids(quantification_cells, keyCellId_valueEdgeId, column_name = 'edge_ids', identifiers = 'initial_id')
    quantification_edges, mask_edges = track_edges_vertices_alternative_approach(mask_edges, quantification_edges, neighbor_configs = 'cell_ids', t_column = 'frame', tracking_id = 'particle')

    ## identify border cells
    border_cell_dictionary = get_cell_boundary_annotation(quantification_edges)
    quantification_cells = map_nested_dictionary_to_ids(quantification_cells, border_cell_dictionary, column_name = 'border_cell')

    ## vertices stuff
    vertices_neighbors_dict = get_dictionary_keyEdgeVertexId_valueCellId(mask_vertices, mask_bckgrdlbl)
    quantification_vertices = map_nested_dictionary_to_ids(quantification_vertices, vertices_neighbors_dict, column_name = 'cell_ids', identifiers = 'initial_id')
    quantification_vertices = map_cellId_to_cellAnno(quantification_vertices, quantification_cells)
    keyCellId_valueVertexId = get_dictionary_keyCellId_valueEdgeVertexId(quantification_vertices, identifiers = 'initial_id')
    quantification_cells = map_nested_dictionary_to_ids(quantification_cells, keyCellId_valueVertexId, column_name = 'vertex_ids',  identifiers = 'initial_id')
    quantification_vertices, mask_vertices = track_edges_vertices_alternative_approach(mask_vertices, quantification_vertices, neighbor_configs = 'cell_ids', t_column = 'frame', tracking_id = 'particle')

    ## getting signal values
    quantification_cells = get_signal_quantification(image, mask, quantification_cells)
    quantification_edges = get_signal_quantification(image, mask_edges, quantification_edges)
    quantification_vertices = get_signal_quantification(image, mask_vertices, quantification_vertices)

    ##### SAVE ALL THE DATA
    save_data(viewer, global_plugin_data, quantification_cells, data_type = 'table', file_path_default = file_path_quantification_cells, message = "Save the quantification of the cells to disk")
    save_data(viewer, global_plugin_data, quantification_edges, data_type = 'table', file_path_default = "quantification_edges.tsv", message = "Save the quantification of edges to disk")    
    save_data(viewer, global_plugin_data, quantification_vertices, data_type = 'table', file_path_default = "quantification_vertices.tsv", message = "Save the quantification of vertices to disk")    

    ## Saving the masks
    mask_bckgrdlbl[mask_bckgrdlbl != 999 ] = 0
    file_path = save_data(viewer, global_plugin_data, mask_bckgrdlbl, data_type = 'label', file_path_default = "mask_bckgrdlbl.tif", message = "Save the mask_bckgrdlbl to disk")    
    if file_path is not None:
        viewer.add_labels(mask_bckgrdlbl, name=os.path.basename(file_path)) ## update the viewer with the new mask

    file_path = save_data(viewer, global_plugin_data, mask_edges, data_type = 'label', file_path_default = "mask_edges.tif", message = "Save the relabeled mask_edges to disk")    
    if file_path is not None:
        viewer.add_labels(mask_edges, name=os.path.basename(file_path)) ## update the viewer with the new mask

    file_path = save_data(viewer, global_plugin_data, mask_vertices, data_type = 'label', file_path_default = "mask_vertices.tif", message = "Save the relabeled mask_vertices to disk")    
    if file_path is not None:
        viewer.add_labels(mask_vertices, name=os.path.basename(file_path)) ## update the viewer with the new mask

    # centroids id and saving
    mask_edges_ids = create_centroid_particleIDs_stack(mask_edges, quantification_edges, font_size = centroid_font_size)
    file_path = save_data(viewer, global_plugin_data, mask_edges_ids, data_type = 'label', file_path_default = "mask_edges_ids.tif", message = "Save the mask with centroids as particle ID to disk")    
    if file_path is not None:
        viewer.add_labels(mask_edges_ids, name=os.path.basename(file_path)) ## update the viewer with the new mask
    # centroids id and saving
    mask_vertices_ids = create_centroid_particleIDs_stack(mask_vertices, quantification_vertices, font_size = centroid_font_size)
    file_path = save_data(viewer, global_plugin_data, mask_vertices_ids, data_type = 'label', file_path_default = "mask_vertices_ids.tif", message = "Save the mask with centroids as particle ID to disk")    
    if file_path is not None:
        viewer.add_labels(mask_vertices_ids, name=os.path.basename(file_path)) ## update the viewer with the new mask
    
    print("Quantification of edges and vertices successfully finished")




@magic_factory(
    call_button="Export Custom Table",
    quantification_cells={"choices": global_plugin_data.keys()})
def export_custom_table(viewer: napari.Viewer, 
                        quantification_cells: str):
    
    quantification_cells = global_plugin_data[quantification_cells]

    included_columns_widget = Select(
        choices=list(quantification_cells.columns),
        label="Included Columns",
        allow_multiple=True
    )
    included_columns_widget.value = list(quantification_cells.columns)

    query_widget = LineEdit(
        label="Query fields",
        value="annotation == 'DE' and frame > 2"
    )

    melt_widget = Select(
        value = None, 
        choices=['frame', 'particle', 'annotation'],
        label="Melt",
        allow_multiple=False,
        nullable = True
    )

    group_by_widget = Select(
        value = None,
        choices=['frame', 'particle', 'annotation'],
        label="Group by",
        allow_multiple=False, 
        nullable = True
    )

    def export_action():
        included_columns = included_columns_widget.value
        query_by = query_widget.value
        melt_by = melt_widget.value
        group_by = group_by_widget.value
        print(f'Included columns: {included_columns}')
        print(f'Query string: {query_by}')
        print(f'Melt by: {melt_by}')
        print(f'Group by: {group_by}')
        custom_table = create_custom_table(quantification_cells, included_columns, query_by, melt_by, group_by)

        print(f'Custom table datatype: {type(custom_table)}')
        save_data(viewer, global_plugin_data, custom_table, data_type = 'table', file_path_default = f'custom_table', message = "Save the custom table to disk")

    export_button = PushButton(text='Create Custom Table')
    export_button.clicked.connect(export_action)

    container = Container(widgets=[
        included_columns_widget,
        query_widget,
        melt_widget,
        group_by_widget,
        export_button
    ])

    viewer.window.add_dock_widget(container, name='Custom Table Exporter', area='right')


@magic_factory(call_button="Load Initial Mask and Quantification")
def assign_populations(viewer: napari.Viewer,
                       name_initial_pop: str = "initial_population",
                       ):

    # Load mask and table
    message = "Load a mask containing that you want to assign into distinct populations"
    mask_layer, initial_mask, file_path_mask = load_data(viewer, global_plugin_data, data_type='label', message=message)
    mask_layer.name = name_initial_pop
    file_path_initial_mask = os.path.join(os.path.dirname(file_path_mask), name_initial_pop + ".tif")

    message = "Load the quantification table that is associated with this mask"
    initial_quant_table, file_path_quant_table = load_data(viewer, global_plugin_data, data_type='table', message=message)
    file_path_initial_quant_table = os.path.join(os.path.dirname(file_path_mask), name_initial_pop + "_quantification.tsv")

    # Shared mutable state
    state = {
        "first_iteration": True,
    }

    @magicgui(call_button="Save Population and Initiate New Population")
    def save_and_initiate(name_new_pop: str = "new_population"):
        if not state["first_iteration"]:
            # Save existing
            save_data(viewer, global_plugin_data, state["new_mask"], data_type='label', file_path_default=state["file_path_new_mask"])
            save_data(viewer, global_plugin_data, state["new_quant_table"], data_type='table', file_path_default=state["file_path_new_quant_table"])
            save_data(viewer, global_plugin_data, initial_mask, data_type='label', file_path_default=file_path_initial_mask)
            save_data(viewer, global_plugin_data, initial_quant_table, data_type='table', file_path_default=file_path_initial_quant_table)

        # Reset for next population
        state["new_mask"] = np.zeros_like(initial_mask)
        state["new_quant_table"] = pd.DataFrame(columns=initial_quant_table.columns)
        state["file_path_new_mask"] = os.path.join(os.path.dirname(file_path_mask), name_new_pop + ".tif")
        state["file_path_new_quant_table"] = os.path.join(os.path.dirname(file_path_mask), name_new_pop + "_quantification.tsv")
        new_layer = viewer.add_labels(state["new_mask"], name=name_new_pop)
        state["new_layer"] = new_layer
        state["first_iteration"] = False

        viewer.layers.selection.active = mask_layer

    def assign_entity(layer, event):
        coords = tuple(np.round(event.position).astype(int))
        if len(coords) != 3:
            print("Unsupported dimensionality.")
            return
        z, y, x = coords
        particle_id = initial_mask[z, y, x]
        if particle_id == 0:
            print("Clicked background.")
            return

        print(f"Clicked particle: {particle_id}")
        state["new_mask"][initial_mask == particle_id] = particle_id
        initial_mask[initial_mask == particle_id] = 0
        mask_layer.data = initial_mask ## LAST CHANGE SEE IF IT DOES THE JOB
        state['new_layer'].data = state["new_mask"]
        mask_layer.data = initial_mask

        # Transfer table rows
        selected_rows = initial_quant_table[initial_quant_table["particle"] == particle_id]
        state["new_quant_table"] = pd.concat([state["new_quant_table"], selected_rows])

    # Connect click callback
    mask_layer.mouse_drag_callbacks.append(assign_entity)
    viewer.window.add_dock_widget(save_and_initiate, area='right')





