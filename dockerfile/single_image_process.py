
from torchvision  import transforms


class ImageProcessor:
    def __init__(self):
        """
        The function takes in an image, resizes it to 128x128, crops it to 128x128, flips it
        horizontally with a probability of 0.3, converts it to a tensor, and normalizes it
        """
        self.transform = transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(self.repeat_channel),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    
    def repeat_channel(x):
            return x.repeat(3, 1, 1)

    def __call__(self, image):
        """
        It takes an image, checks if it's in RGB mode, and if it is, it transforms it to grayscale. If
        it's already in grayscale, it just transforms it
        
        :param image: The image to be transformed
        :return: A numpy array of the image with a dimension added to the front.
        """
        if image.mode != 'RGB':
            image = self.transform_Gray(image)
        else:
            image = self.transform(image)
        
        # Add a dimension to the image
        image = image[None, :, :, :]
        return image

