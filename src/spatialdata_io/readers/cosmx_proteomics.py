from __future__ import annotations

import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable

import dask.array as da
import numpy as np
import pandas as pd
import pyarrow as pa
from anndata import AnnData
from dask_image.imread import imread
from scipy.sparse import csr_matrix
from skimage.transform import estimate_transform
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, TableModel
from spatialdata.transformations.transformations import Affine, Identity

from spatialdata_io._constants._constants import CosmxKeys, CosmxProteomicsKeys
from spatialdata_io._docs import inject_docs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dask.dataframe import DataFrame as DaskDataFrame

__all__ = ["cosmx_proteomics"]

def find_files(directory: Path, pattern: str) -> Iterable[Path]:
    for dirpath_str, dirnames, filenames in os.walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.match(pattern):
                yield filepath

def find_files_full_path(directory: Path, pattern: str) -> Iterable[Path]:
    for dirpath_str, dirnames, filenames in os.walk(directory):
        dirpath = Path(dirpath_str)
        for filename in filenames:
            filepath = dirpath / filename
            if filepath.full_match(pattern):
                yield filepath

def find_directory(dir_name:str)->Path:
    pass

def read_plex_text(dir_name:str)->pd.DataFrame:
    plex_text_pattern = "plex*.txt"
    plex_text_file = find_files(find_directory(dir_name), plex_text_pattern)
    plex_text_df = pd.read_csv(plex_text_file)
    plex_text_mapping = {plex_text_df.at[i, 'ProbeID']:plex_text_df.at[i, 'DisplayName'] for i in plex_text_df.index}
    return plex_text_mapping

@inject_docs(cx=CosmxKeys)
def cosmx_proteomics(
    path: str | Path,
    dataset_id: str | None = None,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> SpatialData:
    """Read *Cosmx Nanostring* data.

    This function reads the following files:

        - ``<dataset_id>_`{cx.COUNTS_SUFFIX!r}```: Counts matrix.
        - ``<dataset_id>_`{cx.METADATA_SUFFIX!r}```: Metadata file.
        - ``<dataset_id>_`{cx.FOV_SUFFIX!r}```: Field of view file.
        - ``{cx.IMAGES_DIR!r}``: Directory containing the images.
        - ``{cx.LABELS_DIR!r}``: Directory containing the labels.

    .. seealso::

        - `Nanostring Spatial Molecular Imager <https://nanostring.com/products/cosmx-spatial-molecular-imager/>`_.

    Parameters
    ----------
    path
        Path to the root directory containing *Nanostring* files.
    dataset_id
        Name of the dataset.
    imread_kwargs
        Keyword arguments passed to :func:`dask_image.imread.imread`.
    image_models_kwargs
        Keyword arguments passed to :class:`spatialdata.models.Image2DModel`.

    Returns
    -------
    :class:`spatialdata.SpatialData`
    """
    path = Path(path)

    # tries to infer dataset_id from the name of the counts file

    if dataset_id is None:
        counts_files = find_files(path, f"*{CosmxProteomicsKeys.COUNTS_SUFFIX}*")
        if len(counts_files) == 1:
            found = re.match(rf"(.*)_{CosmxProteomicsKeys.COUNTS_SUFFIX}*", counts_files[0].name)
            if found:
                dataset_id = found.group(1)
    if dataset_id is None:
        raise ValueError("Could not infer `dataset_id` from the name of the counts file. Please specify it manually.")

    # check for file existence
    counts_file = find_files(path, f"*{CosmxProteomicsKeys.COUNTS_SUFFIX}*")[0]
    if not counts_file.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}.")
    meta_file = find_files(path, f"{dataset_id}_{CosmxProteomicsKeys.METADATA_SUFFIX}*")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}.")
    fov_file = find_files(path, f"{dataset_id}_{CosmxProteomicsKeys.FOV_SUFFIX}*")
    if not fov_file.exists():
        raise FileNotFoundError(f"Found field of view file: {fov_file}.")
    images_dir = path / CosmxProteomicsKeys.IMAGES_DIR
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}.")
    labels_dir = path / CosmxProteomicsKeys.LABELS_DIR
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}.")

    counts = pd.read_csv(path / counts_file, header=0, index_col=CosmxProteomicsKeys.INSTANCE_KEY)
    counts.index = counts.index.astype(str).str.cat(counts.pop(CosmxProteomicsKeys.FOV).astype(str).values, sep="_")

    obs = pd.read_csv(path / meta_file, header=0, index_col=CosmxProteomicsKeys.INSTANCE_KEY)
    obs[CosmxProteomicsKeys.FOV] = pd.Categorical(obs[CosmxProteomicsKeys.FOV].astype(str))
    obs[CosmxProteomicsKeys.REGION_KEY] = pd.Categorical(obs[CosmxProteomicsKeys.FOV].astype(str).apply(lambda s: s + "_labels"))
    obs[CosmxProteomicsKeys.INSTANCE_KEY] = obs.index.astype(np.int64)
    obs.rename_axis(None, inplace=True)
    obs.index = obs.index.astype(str).str.cat(obs[CosmxProteomicsKeys.FOV].values, sep="_")

    common_index = obs.index.intersection(counts.index)

    adata = AnnData(
        csr_matrix(counts.loc[common_index, :].values),
        dtype=counts.values.dtype,
        obs=obs.loc[common_index, :],
    )
    adata.var_names = counts.columns

    table = TableModel.parse(
        adata,
        region=list(set(adata.obs[CosmxProteomicsKeys.REGION_KEY].astype(str).tolist())),
        region_key=CosmxProteomicsKeys.REGION_KEY.value,
        instance_key=CosmxProteomicsKeys.INSTANCE_KEY.value,
    )

    fovs_counts = list(map(str, adata.obs.fov.astype(int).unique()))

    affine_transforms_to_global = {}

    for fov in fovs_counts:
        idx = table.obs.fov.astype(str) == fov
        loc = table[idx, :].obs[[CosmxProteomicsKeys.X_LOCAL_CELL, CosmxProteomicsKeys.Y_LOCAL_CELL]].values
        glob = table[idx, :].obs[[CosmxProteomicsKeys.X_GLOBAL_CELL, CosmxProteomicsKeys.Y_GLOBAL_CELL]].values
        out = estimate_transform(ttype="affine", src=loc, dst=glob)
        affine_transforms_to_global[fov] = Affine(
            # out.params, input_coordinate_system=input_cs, output_coordinate_system=output_cs
            out.params,
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        )

    table.obsm["global"] = table.obs[[CosmxProteomicsKeys.X_GLOBAL_CELL, CosmxProteomicsKeys.Y_GLOBAL_CELL]].to_numpy()
    table.obsm["spatial"] = table.obs[[CosmxProteomicsKeys.X_LOCAL_CELL, CosmxProteomicsKeys.Y_LOCAL_CELL]].to_numpy()
    table.obs.drop(
        columns=[CosmxProteomicsKeys.X_LOCAL_CELL, CosmxProteomicsKeys.Y_LOCAL_CELL, CosmxProteomicsKeys.X_GLOBAL_CELL, CosmxProteomicsKeys.Y_GLOBAL_CELL],
        inplace=True,
    )

    # prepare to read images and labels
    file_extensions = (".jpg", ".png", ".jpeg", ".tif", ".tiff")
    pat = re.compile(r".*_F(\d+)")

    # check if fovs are correct for images and labels
    fovs_images = []
    for fname in os.listdir(path / CosmxProteomicsKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fovs_images.append(str(int(pat.findall(fname)[0])))

    fovs_labels = []
    for fname in os.listdir(path / CosmxProteomicsKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fovs_labels.append(str(int(pat.findall(fname)[0])))

    fovs_images_and_labels = set(fovs_images).intersection(set(fovs_labels))
    fovs_diff = fovs_images_and_labels.difference(set(fovs_counts))
    if len(fovs_diff):
        logger.warning(
            f"Found images and labels for {len(fovs_images)} FOVs, but only {len(fovs_counts)} FOVs in the counts file.\n"
            + f"The following FOVs are missing: {fovs_diff} \n"
            + "... will use only fovs in Table."
        )

    channel_mapping = read_plex_text(path)
    num_channels = len(channel_mapping)

    # read images
    images = {}
    for fname in os.listdir(path / CosmxProteomicsKeys.IMAGES_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                num_dims = imread(path / CosmxProteomicsKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze().shape
                multi_channel_img = np.zeroes((num_channels, num_dims[0], num_dims[1]))
                for i, channel in enumerate(channel_mapping):
                    img_path_template = f"**/{fov}/*/ProteinImages/*{channel}"
                    img_path = find_files_full_path(path, img_path_template)
                    multi_channel_img[i] = imread(img_path, **imread_kwargs).squeeze()

                flipped_im = da.flip(multi_channel_img, axis=0)
                parsed_im = Image2DModel.parse(
                    flipped_im,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_image": aff,
                    },
                    dims=("c", "y", "x"),
                    c_coords = list(channel_mapping.values()),
                    rgb=None,
                    **image_models_kwargs,
                )
                images[f"{fov}_image"] = parsed_im
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping image {fname}.")

    # read labels
    labels = {}
    for fname in os.listdir(path / CosmxKeys.LABELS_DIR):
        if fname.endswith(file_extensions):
            fov = str(int(pat.findall(fname)[0]))
            if fov in fovs_counts:
                aff = affine_transforms_to_global[fov]
                num_dims = imread(path / CosmxProteomicsKeys.IMAGES_DIR / fname, **imread_kwargs).squeeze().shape
                multi_channel_mask = np.zeroes((num_channels, num_dims[0], num_dims[1]))
                for i, channel in enumerate(channel_mapping):
                    label_path_template = f"**/{fov}/*/ProteinMasks/*{channel}"
                    label_path = find_files_full_path(path, label_path_template)
                    multi_channel_mask[i] = imread(label_path, **imread_kwargs).squeeze()

                flipped_la = da.flip(multi_channel_mask, axis=0)
                parsed_la = Labels2DModel.parse(
                    flipped_la,
                    transformations={
                        fov: Identity(),
                        "global": aff,
                        "global_only_image": aff,
                    },
                    dims=("c", "y", "x"),
                    c_coords = list(channel_mapping.values()),
                    rgb=None,
                    **image_models_kwargs,
                )
                labels[f"{fov}_labels"] = parsed_la
            else:
                logger.warning(f"FOV {fov} not found in counts file. Skipping labels {fname}.")


    # TODO: what to do with fov file?
    # if fov_file is not None:
    #     fov_positions = pd.read_csv(path / fov_file, header=0, index_col=CosmxKeys.FOV)
    #     for fov, row in fov_positions.iterrows():
    #         try:
    #             adata.uns["spatial"][str(fov)]["metadata"] = row.to_dict()
    #         except KeyError:
    #             logg.warning(f"FOV `{str(fov)}` does not exist, skipping it.")
    #             continue

    return SpatialData(images=images, labels=labels, table=table)
