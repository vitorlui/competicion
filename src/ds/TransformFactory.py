from torchvision import transforms
from torchvision.transforms import InterpolationMode

class TransformFactory:
    MEAN = [0.4603, 0.3696, 0.3388]
    STD = [0.2246, 0.2059, 0.1994]

    @staticmethod
    def get_transform_by_name(name):
        for transform_name, tfms in TransformFactory.get_all_transforms():
            if transform_name == name:
                return transform_name, tfms
        raise ValueError(f"Transform not found: {name}")

    @staticmethod
    def get_all_transforms():
        resize_sizes = [256, 112]
        crop_options = [True]
        normalize_options = [True]
        interpolation_modes = {
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC
        }

        transform_list = []

        for resize in resize_sizes:
            for crop in crop_options:
                for normalize in normalize_options:
                    for interp_name, interp_mode in interpolation_modes.items():
                        name_parts = [f"trans_res{resize}{interp_name}"]
                        tfms = []

                        # Preprocessing
                        tfms.append(transforms.Resize(resize, interpolation=interp_mode))
                        if crop:
                            tfms.append(transforms.CenterCrop(224))
                            name_parts.append("crop224")
                        else:
                            name_parts.append("nocrop")

                        # Postprocessing
                        tfms.append(transforms.ToTensor())
                        if normalize:
                            tfms.append(transforms.Normalize(mean=TransformFactory.MEAN, std=TransformFactory.STD))
                            # name_parts.append("normTrue")
                        # else:
                            # name_parts.append("normFalse")

                        transform_name = "_".join(name_parts)
                        composed = transforms.Compose(tfms)

                        transform_list.append((transform_name, composed))

        return transform_list

    @staticmethod
    def get_preprocessing_transform(resize=256, crop=True, interpolation=InterpolationMode.BILINEAR):
        tfms = [transforms.Resize(resize, interpolation=interpolation)]
        if crop:
            tfms.append(transforms.CenterCrop(224))
        return transforms.Compose(tfms)

    @staticmethod
    def get_postprocessing_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=TransformFactory.MEAN, std=TransformFactory.STD)
        ])