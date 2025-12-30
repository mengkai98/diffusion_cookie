from torchvision import transforms


class CustomResizeCrop:
    def __init__(self, target_size=(1024, 1024)):
        self.target_size = target_size

    def __call__(self, img):
        original_width, original_height = img.size
        scale_ratio = max(self.target_size[0] / original_width, self.target_size[1] / original_height)

        if scale_ratio > 1:
            new_size = (int(original_height * scale_ratio), int(original_width * scale_ratio))
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        new_size,
                    ),
                    transforms.CenterCrop(self.target_size),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.target_size,
                    ),
                    transforms.CenterCrop(self.target_size),
                ]
            )

        return transform(img)
