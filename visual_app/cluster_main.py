# imports
import face_recognition
import cv2
import pickle
from .face_clustering.annoy_face_clustering import AnnoyClustering


def encode_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(image, model='hog')
    if len(boxes) == 1:
        enc_img = face_recognition.face_encodings(image, boxes)
        d = [{"imagePath": img_path, "encodings": enc, "msg": None} for enc in enc_img]
        return d
    else:
        print(f"{img_path} can't be used for training!, contains multiple faces.")
        d = [{"imagePath": img_path, "encodings": None, "msg": "contains multiple faces"}]
        return d


def find_neighbours(path, name):
    data = encode_image(path)
    page_content = [{'names': name, 'image': data[0]['imagePath'], 'dist': 0, 'msg': data[0]['msg']}]
    if not (data[0]['msg']):
        train_data = pickle.loads(
            open('visual_app/face_clustering/face_encodings/train_lfw_images_v1.pickle', 'rb').read())
        index = AnnoyClustering(train_data)
        index.face_clusters()
        indices_list, distances = index.query(data[0]['encodings'])
        for i, j in zip(indices_list, distances):
            temp_dict = {}
            temp_dict['names'] = i.split('/')[4]
            temp_dict['image'] = i
            temp_dict['dist'] = j
            temp_dict['msg'] = None
            page_content.append(temp_dict)
    return page_content


if __name__ == '__main__':
    data_img = find_neighbours('././media/lfw/Mark_Cuban/Mark_Cuban_0002.jpg', 'Mark_Cuban')
    print(data_img)
    print(bool(not (data_img[0]['msg'])))
    test_data = pickle.loads(open('visual_app/face_clustering/face_encodings/test_lfw_images.pickle', 'rb').read())
    print(test_data[1]['imagePath'])
    # test_images = glob.glob('./media/images/*.*')
    # test_images = [i.replace('\\','/') for i in test_images]
    # print(test_images)
    # test_content = []
    # for i in test_images:
    #     test_dict={}
    #     test_dict['names'] = i.split('/')[3]
    #     test_dict['image'] = i
    #     test_content.append(test_dict)
    # print(test_content)
    # print(face_images.name + '.')
    # print(face_images.face_img.url)
