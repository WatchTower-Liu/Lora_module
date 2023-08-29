import torch 
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms

class DreamBoothDataset(Dataset):

    def __init__(
        self,
        instance_data_root,
        tokenizer,
        size=512,
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.instance_prompt = {}
        self.instance_images_path = {}
        for file in list(Path(instance_data_root).iterdir()):
            if file.suffix == ".txt":
                self.instance_prompt[file.name.split(".")[0]] = file.read_text()
            else:
                self.instance_images_path[file.name.split(".")[0]] = file
        
        self.num_instance_images = len(self.instance_images_path)
        
        self._length = self.num_instance_images


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_name = list(self.instance_prompt.keys())[index % self.num_instance_images]
        instance_image = Image.open(self.instance_images_path[instance_name])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt[instance_name],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

