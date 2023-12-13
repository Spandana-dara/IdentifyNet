"""
Python code to implement spotify's Annoy clustering on face embeddings
"""
import pickle
from annoy import AnnoyIndex


train_data = pickle.loads(open('visual_app/face_clustering/face_encodings/train_lfw_images_v1.pickle', 'rb').read())
test_data = pickle.loads(open('visual_app/face_clustering/face_encodings/test_lfw_images.pickle', 'rb').read())
print(f'No. of unique samples in train_data: {len(train_data)}, No. of samples in test_data: {len(test_data)}')


class AnnoyClustering:
    def __init__(self, encoded_data):
        self.face_embeddings = [d['encodings'] for d in encoded_data]
        self.dimension = len(self.face_embeddings[0])
        self.labels = [d['imagePath'] for d in encoded_data]
        self.annoy_tree = AnnoyIndex(self.dimension, 'euclidean')

    def face_clusters(self, number_of_trees=100):
        for i, embed in enumerate(self.face_embeddings):
            self.annoy_tree.add_item(i, embed.tolist())
        self.annoy_tree.build(number_of_trees)

    def query(self, face, k=3):
        indices = self.annoy_tree.get_nns_by_vector(face.tolist(), k, search_k=-1, include_distances=True)
        return [self.labels[i] for i in indices[0]], indices[1]


if __name__ == "__main__":
    index = AnnoyClustering(train_data)
    index.face_clusters()
    correct = 0
    not_train_face = 0
    train_names = [d['imagePath'].split('/')[4] for d in train_data]
    test_images_not_correct = []
    for i in test_data[0:1]:
        indices_list, distances = index.query(i['encodings'])
        print(indices_list)
        name = i['imagePath'].split('/')[4]
        print(name)
        if name in indices_list:
            correct += 1
        else:
            test_images_not_correct.append(i['imagePath'])
        if name not in train_names:
            not_train_face += 1
    print(f"No of faces not present in training set: {not_train_face}")
    print(f"Accuracy of the model is: {correct*100/(len(test_data)-not_train_face)}")
    print(f"Wrongly clustered images:{len(test_images_not_correct)}")
    print(test_data[0]['imagePath'])



