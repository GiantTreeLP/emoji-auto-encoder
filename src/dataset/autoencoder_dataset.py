import os.path
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Any

import PIL.Image as Image
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

_DUMMY_CLASS = ""


def pil_loader_rgba(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA")


class AutoEncoderDataset(ImageFolder):

    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None,
                 *, repetitions: int = 1):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.classes, self.class_to_idx = self._find_classes(root)
        self.samples = self._augment_samples(self.samples) * repetitions
        self.targets = [s[1] for s in self.samples]

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return [_DUMMY_CLASS], {_DUMMY_CLASS: 0}

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # One category for each image
        classes = []
        path: Path
        for i, path in enumerate(Path(directory).glob("**/*.png")):
            classes.append(path.name)
        return classes, {c: i for i, c in enumerate(classes)}

    def _augment_samples(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        augmented_samples = []
        for path, target in samples:
            class_index = self.class_to_idx[os.path.basename(path)]
            augmented_samples.append((path, class_index))
        return augmented_samples
