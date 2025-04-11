import os
import numpy as np
from src.dataloaders.tiff_utils import save_array_to_tiff
import matplotlib.pyplot as plt


def save_flows_as_tif(
        frame_labels, flows, save_dir, flow_description, crs_meta_data='+proj=latlong', transform_meta_data=None):
    for frame_label, flow in zip(frame_labels, flows):
        save_dir_example = os.path.join(save_dir, frame_label)
        os.makedirs(save_dir_example, exist_ok=True)

        tif_filename = os.path.join(save_dir_example, f"{frame_label}_{flow_description}_ew.tif")
        save_array_to_tiff(flow[0].astype(np.float32), tif_filename, transform=transform_meta_data, crs=crs_meta_data)
        tif_filename = os.path.join(save_dir_example, f"{frame_label}_{flow_description}_ns.tif")
        save_array_to_tiff(flow[1].astype(np.float32), tif_filename, transform=transform_meta_data, crs=crs_meta_data)


def save_flows_as_png(frame_labels, flows, save_dir, flow_description, scale=1):
    for frame_label, flow in zip(frame_labels, flows):
        save_dir_example = os.path.join(save_dir, frame_label)
        os.makedirs(save_dir_example, exist_ok=True)

        ew_png_filename = os.path.join(save_dir_example, f"{frame_label}_{flow_description}_ew.png")
        plt.imsave(ew_png_filename, -flow[0], cmap='seismic', vmin=-scale, vmax=scale)
        ns_png_filename = os.path.join(save_dir_example, f"{frame_label}_{flow_description}_ns.png")
        plt.imsave(ns_png_filename, -flow[1], cmap='seismic', vmin=-scale, vmax=scale)


def save_all_flows_as_png(frame_label, flows, save_dir, flow_description, direction="ns", scale=1):
    save_dir_example = os.path.join(save_dir, frame_label)
    os.makedirs(save_dir_example, exist_ok=True)

    direction_id = 0 if direction=="ew" else 1
    fig, axs = plt.subplots(3, 4, figsize=(12, 6))
    ns_png_filename = os.path.join(save_dir_example, f"{frame_label}_{flow_description}_{direction}_all.png")
    
    for i, ax in enumerate(axs.flat):
        if i >= len(flows):
            break
        im = ax.imshow(-flows[i][direction_id], cmap='seismic', vmin=-scale, vmax=scale)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(ns_png_filename, bbox_inches='tight')
    plt.close()
    print(f"saved {ns_png_filename}")


def apply_convention_and_save(
        frame_labels, flows, save_dir, flow_description, crs_meta_data='+proj=latlong', transform_meta_data=None, scale=1):
    flows_qgis_convention = flows.clone()
    flows_qgis_convention[:, 1] *= -1
    tif_dir = os.path.join(save_dir, "tifs")
    save_flows_as_tif(
        frame_labels, 
        flows_qgis_convention.cpu().numpy(), 
        tif_dir, 
        flow_description, 
        crs_meta_data, 
        transform_meta_data
    )

    # png_dir = os.path.join(save_dir, "pngs")
    # save_flows_as_png(
    #     frame_labels, 
    #     flows_qgis_convention.cpu().numpy(), 
    #     png_dir, 
    #     flow_description, 
    #     scale=scale)

