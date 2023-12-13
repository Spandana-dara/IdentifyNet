# imports
import face_recognition
import pickle
import cv2
import glob
import random
from data_extraction import ImageExtractor

image_extractor = ImageExtractor('././media/lfw/*', 2)
image_train_paths, image_test_paths = image_extractor.extract()
random.shuffle(image_test_paths)
test_img_paths = image_test_paths[:1500]
print(f"Number of unique faces:{len(image_train_paths)}\n"
      f"Total images available for testing: {len(image_test_paths) + len(image_train_paths)}")


class FaceClusters:
    def __init__(self, model='hog', save_encodings=False):
        self.model = model
        self.save_encodings = save_encodings

    def encode_images(self, img_paths, encoding_path=None):
        data = []
        print('Started encoding images!!')
        for (i, img_path) in enumerate(img_paths):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(image, model=self.model)
            if len(boxes) == 1:
                enc_img = face_recognition.face_encodings(image, boxes)
                d = [{"imagePath": img_path, "encodings": enc} for enc in enc_img]
                data.extend(d)
            else:
                print(f"{img_path} can't be used for training!, contains multiple faces.")
                continue
        print(len(data))
        print('Encoded the images!!')
        if self.save_encodings and encoding_path is not None:
            enc_path = f'visual_app/face_clustering/face_encodings/{encoding_path}'
            file = open(enc_path, 'wb')
            file.write(pickle.dumps(data))
            file.close()
            print(f"Access the image encodings from {enc_path}")
            return data, enc_path
        else:
            print("Encodings aren't stored!, if it needs to be stored then do it afterwards.")
            return data, encoding_path


if __name__ == "__main__":
    face_clusters = FaceClusters('hog', True)
    print(glob.glob('/media/lfw/*'))
    train_data, _ = face_clusters.encode_images(image_train_paths, 'train_lfw_images_v1.pickle')
    # test_data, _ = face_clusters.encode_images(image_test_paths, 'test_lfw_images.pickle')
    # train_data = pickle.loads(open('face_encodings/train_lfw2_images.pickle', 'rb').read())
    # test_data = pickle.loads(open('face_encodings/test_lfw2_images.pickle', 'rb').read())
    print(f'train_data: {len(train_data)}')
