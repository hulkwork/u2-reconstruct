import os
from glob import glob
from typing import Dict, List, Tuple

from PIL import Image

from src.datasets.dataset import FilesDataset


class AbnormalDataset(FilesDataset):
    def __init__(
        self,
        imgs_files: List[str],
        gt_path: str,
        resize: int = 256,
        normalized: bool = True,
        metadata: Dict[str, str] = None,
    ):
        super().__init__(imgs_files, resize, normalized, metadata)
        self.gt_path = gt_path

    def __getitem__(self, i):
        result = super().__getitem__(i)
        img_file = self.ids[i]
        basename = os.path.basename(img_file)
        filename = f"{os.path.splitext(basename)[0]}_mask.png"
        gt_img = Image.open(
            os.path.join(self.gt_path, self.metadata["status"], filename)
        )
        result["gt"] = self.preprocess(
            gt_img, new_size=self.resize, normalize=self.normalize
        )

        return result


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
                metadata={"item": item},
                is_transfo=True
            ),
            test=FilesDataset(
                imgs_files=glob(test_data + "/*.png"),
                resize=resized,
                normalized=normalized,
                metadata={"item": item},
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
            item_abnormal[item][status] = AbnormalDataset(
                imgs_files=glob(path),
                resize=resized,
                normalized=normalized,
                metadata={"status": status, "item": item},
                gt_path=gt_item_status_path,
            )
            gt_defect[item][status] = FilesDataset(
                imgs_files=glob(os.path.join(gt_item_status_path, status, "*.png")),
                resize=resized,
                normalized=normalized,
                metadata={"status": status, "item": item},
            )

    return datasets, item_abnormal, gt_defect
