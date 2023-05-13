from PIL import Image
from io import BytesIO
from torch import nn


class JPEGCompression(object):
    def __init__(self, quality):
        self.quality = quality
        self.num_images = 0.0001
        self.total_bytes = 0

    def average_size(self):
        return 1.0 * self.total_bytes / self.num_images

    def __call__(self, image):
        self.num_images += 1
        if self.quality is None:
            self.total_bytes += image.size[0] * image.size[0] * 3
            return image
        else:
            outputIoStream = BytesIO()
            image.save(outputIoStream, "JPEG", quality=self.quality, optimice=True)
            self.total_bytes += outputIoStream.getbuffer().nbytes
            outputIoStream.seek(0)
            return Image.open(outputIoStream)
