import os
from glob import glob
from typing import Dict, Tuple

from src.datasets.dataset import FilesDataset


def load_mvtec_dataset(
    data_path: str = "./data/", resized: int = 256, normalized: bool = False
) -> Tuple[
    Dict[str, FilesDataset],
    Dict[str, Dict[str, FilesDataset]],
    Dict[str, Dict[str, FilesDataset]],
]:
    all_items = os.listdir(data_path)
    all_items = filter(lambda x: x not in ["license.txt", "readme.txt"], all_items)
    test_path = "test"
    train_path = "train"
    datasets = dict()
    item_abnormal = {}
    gt_defect = {}
    for item in all_items:
        train_data = os.path.join(data_path, item, train_path, "good")
        test_data = os.path.join(data_path, item, test_path, "good")

        datasets[item] = dict(
            train=FilesDataset(
                imgs_files=glob(train_data + "/*.png"),
                resize=resized,
                normalized=normalized,
            ),
            test=FilesDataset(
                imgs_files=glob(test_data + "/*.png"),
                resize=resized,
                normalized=normalized,
            ),
        )
        item_abnormal[item] = {}
        gt_defect[item] = {}
        desc_item_status_path = os.path.join(data_path, item, test_path)
        gt_item_status_path = os.path.join(data_path, item, "ground_truth")
        desc_item_status = filter(
            lambda x: x != "good", os.listdir(desc_item_status_path)
        )
        for status in desc_item_status:
            path = os.path.join(desc_item_status_path, status, "*.png")
            item_abnormal[item][status] = FilesDataset(
                imgs_files=glob(path),
                resize=resized,
                normalized=normalized,
            )
            gt_defect[item][status] = FilesDataset(
                imgs_files=glob(os.path.join(gt_item_status_path, status, "*.png")),
                resize=resized,
                normalized=normalized,
            )

    return datasets, item_abnormal, gt_defect
