"""
Simple python class to extract all the image paths from folders containing atleast 10 files
input: dir_path containing directories of images
output: array of image paths
"""
import glob


class ImageExtractor:
    def __init__(self, main_dir, min_images=10):
        self.DIR = main_dir
        self.min_images = min_images

    def extract(self):
        path_lfw = [i.replace("\\", '/') for i in glob.glob(self.DIR)]
        image_train_paths = []
        image_test_paths = []
        n_classes = 0
        for path in path_lfw:
            no_images = [i.replace('\\', '/') for i in glob.glob(path + '/*')]
            if len(no_images) >= self.min_images:
                n_classes += 1
                image_train_paths.append(no_images[0])
                image_test_paths.extend(no_images[1:])
            else:
                continue
        print(f"number of classes present in the data: {n_classes}")
        return image_train_paths, image_test_paths


if __name__ == '__main__':
    image_extractor = ImageExtractor('././media/lfw/*', 2)
    img_train_paths, img_test_paths = image_extractor.extract()
    # print(img_train_paths)
    print(f"Number of unique faces:{len(img_train_paths)}\nTotal images available for testing:"
          f"{len(img_test_paths)} + {len(img_train_paths)}")
    all_names = [i.split('/')[4] for i in img_train_paths]
    local_names = ['Hemanth_Bodapati','Mateen_Sofi','Sai_Shiva_Kalyan_Challa','Ahad_Hamirani']
    print(local_names[0] in all_names)
    print(local_names[1] in all_names)
    print(local_names[2] in all_names)
    print(local_names[3] in all_names)
