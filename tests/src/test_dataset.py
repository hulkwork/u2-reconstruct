import unittest 
import os 
import shutil 
import numpy as np
import cv2

from src.dataset import ReconstructDataset

class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = "./data_test"
        self.nb_images = 10
        os.makedirs(self.data_dir, exist_ok=True)
        for i in range(self.nb_images):
            
            img = np.random.rand(56,56,3) * 255
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(self.data_dir, f"{i}.png"), img)

        return super().setUp()
    def tearDown(self) -> None:
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir, ignore_errors=True)

        return super().tearDown()
    def test_reconstruct(self):
        obj = ReconstructDataset(imgs_dir=self.data_dir)
        self.assertEqual(obj[0]["image"].shape, obj[0]["output"].shape)
        
    
if __name__ == "__main__":
    unittest.main()