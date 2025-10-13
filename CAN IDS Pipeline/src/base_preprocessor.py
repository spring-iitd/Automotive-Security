import os
from abc import ABC, abstractmethod
import shutil   

class DataPreprocessor(ABC):
    def run(self, dataset_path, **kwargs):
        files = os.listdir(dataset_path)
        for file in files : 
            full_file_path = os.path.join(dataset_path,file)
            if os.path.isdir(full_file_path) or file.endswith('.py'):
                continue
            self._move_to_original(full_file_path)
        orig_file_path = os.path.join(dataset_path,"original_dataset")
        modified_file_path = self._get_modified_dataset_path(dataset_path)
        self.preprocess_dataset(orig_file_path, modified_file_path, **kwargs)

    def _move_to_original(self, file_path):
        dir_path = os.path.dirname(file_path)
        orig_dir_path = os.path.join(dir_path, "original_dataset")
        os.makedirs(orig_dir_path, exist_ok=True)
        filename = os.path.basename(file_path)
        dest_path = os.path.join(orig_dir_path, filename)
        shutil.move(file_path, dest_path)  

    def _get_modified_dataset_path(self, dataset_path):
        mod_dir_path = os.path.join(dataset_path, "modified_dataset")
        os.makedirs(mod_dir_path, exist_ok=True)

        return mod_dir_path

    @abstractmethod
    def preprocess_dataset(self, orig_file_path, modified_file_path, **kwargs):
        """
        User should read from "orig_file_path" and save to "modified_file_path"
        """
        pass
