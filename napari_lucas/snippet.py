

@magic_factory(
    call_button="Export Custom Table",
    quantification_table={"choices": global_plugin_data.keys()})
def export_custom_table(viewer: napari.Viewer, 
                        quantification_table: str):
    
    quantification_table = global_plugin_data[quantification_table]

    included_columns_widget = Select(
        choices=list(quantification_table.columns),
        label="Included Columns",
        allow_multiple=True
    )
    included_columns_widget.value = list(quantification_table.columns)

    
    group_by_widget = Select(
        choices=list(quantification_table.select_dtypes(include=['category', 'object']).columns) + [False],
        label="Group by",
        allow_multiple=False
    )

    def export_action():
        included_columns = included_columns_widget.value
        group_by = group_by_widget.value
        custom_table = create_custom_table(quantification_table, included_columns, group_by)

        save_data(viewer, global_plugin_data, custom_table, data_type = 'table', file_path_default = f'custom_table', message = "Save the custom table to disk")


    export_button = PushButton(text='Create Custom Table')
    export_button.clicked.connect(export_action)

    container = Container(widgets=[
        included_columns_widget,
        group_by_widget,
        export_button
    ])

    viewer.window.add_dock_widget(container, name='Custom Table Exporter', area='right')




### snippet 6/25/2025


@magic_factory(call_button="Start Annotation / Switch Annotation Value", annotation={"choices": ["DE", "VE", "exVE", "other"]})
def annotate_cells(viewer: "napari.Viewer",
                   mask_layer: "napari.layers.Labels",
                   annotation: str = "DE"):

    ## RESET POINT

    ## create the other_layer and set as active
    other_layer = viewer.add_labels(mask_layer.data, name = 'other')
    viewer.layers.selection.active = other_layer

    # set the active layer to the selected one
    viewer.layers.selection.active = mask_layer

    # Persistent state stored in the layer metadata (avoids reloading)
    if 'quant_table' not in mask_layer.metadata:
        mask_layer.metadata['quant_table'],  mask_layer.metadata['file_path'] = load_data(viewer, global_plugin_data, data_type = 'table', message = 'Load the quantification table that you want to annotate')

    # Update current annotation state
    mask_layer.metadata['current_annotation'] = annotation
    print(f"Annotation mode set to '{annotation}'.")

    # Define the click callback if not already defined
    if 'callback_connected' not in mask_layer.metadata:

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

            print(f"Annotated particle {particle_id} as '{current_annotation}'.")

            # Refresh display
            layer.refresh()

        # Clear previous callbacks to ensure only one is active
        mask_layer.mouse_drag_callbacks.clear()
        mask_layer.mouse_drag_callbacks.append(click_callback)
        mask_layer.metadata['callback_connected'] = True
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
        quant_table = widget.mask_layer.value.metadata.get('quant_table', None)
        file_path =  widget.mask_layer.value.metadata.get('file_path', '')
        viewer = widget.viewer.value
        if (quant_table is None) or (file_path is None):
            print("Mask_layer is either none or missing quant_table or file_path")
            return

        save_data(
            viewer, 
            global_plugin_data,
            data=quant_table, 
            data_type='table', 
            file_path_default=file_path, 
            message='Saving quantification table with updated annotation')

    save_button.clicked.connect(on_save_button_clicked)
    # Add save button below the widget
    layout.addWidget(save_button)

    container_widget.setLayout(layout)
    return container_widget