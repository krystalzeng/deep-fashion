import tensorflow as tf
import numpy as np
import scipy.ndimage as scim
import os
import pickle

flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator:
    def __init__(self, dataset, images, testset_proportion=0.0):
        """
        Dataset shape: list of tuple of the form:
        (image_filename_1, image_filename_2, type_1, type_2, label)
        Images: dict with key=image_filename, value=nd array with shape (image_width, image_height, nb_channels)
        """
        np.random.seed(0)

        self.matches_dataset = [d for d in dataset if d[4] == 1]
        self.nonmatches_dataset = [d for d in dataset if d[4] == 0]

        self.matches_test = self.matches_dataset[:int(len(self.matches_dataset) * testset_proportion)]
        self.matches_dataset = self.matches_dataset[int(len(self.matches_dataset) * testset_proportion):]

        self.nonmatches_test = self.nonmatches_dataset[:int(len(self.nonmatches_dataset) * testset_proportion)]
        self.nonmatches_dataset = self.nonmatches_dataset[int(len(self.nonmatches_dataset) * testset_proportion):]

        print("Matches count: %d" % (len(self.matches_dataset)))
        print("Non-matches count: %d" % (len(self.nonmatches_dataset)))

        self.images = images

        self.matches_batch_index = 0
        self.nonmatches_batch_index = 0

        # The indices to traverse the matches dataset and the non matches dataset.
        # After all the batches have been generated for one epoch, the indices are reshuffled.
        self.match_indices = np.random.choice(range(0, len(self.matches_dataset)),
                                              size=(len(self.matches_dataset)), replace=True)
        self.nonmatch_indices = np.random.choice(range(0, len(self.nonmatches_dataset)),
                                                 size=(len(self.nonmatches_dataset)), replace=True)

    def train_set(self):
        left = []
        right = []
        labels = []

        for match in self.matches_dataset:
            left_title = match[0]
            right_title = match[1]
            label = match[4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])
            labels.append(label)

        for nonmatch in self.nonmatches_dataset:
            left_title = nonmatch[0]
            right_title = nonmatch[1]
            label = nonmatch[4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])
            labels.append(label)

        left = np.array(left)
        right = np.array(right)
        labels = np.array(labels).reshape((-1, 1))
        return left, right, labels

    def test_set(self):
        left = []
        right = []
        labels = []

        for match in self.matches_test:
            left_title = match[0]
            right_title = match[1]
            label = match[4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])
            labels.append(label)

        for nonmatch in self.nonmatches_test:
            left_title = nonmatch[0]
            right_title = nonmatch[1]
            label = nonmatch[4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])
            labels.append(label)

        left = np.array(left)
        right = np.array(right)
        labels = np.array(labels).reshape((-1, 1))
        return left, right, labels

    def next_batch(self, batch_size, matches_proportion):
        # If we've passed through all the instances, reshuffle the data
        matches_count = int(matches_proportion * batch_size)
        nonmatches_count = int((1 - matches_proportion) * batch_size)

        left = []
        right = []
        labels = []

        if self.matches_batch_index >= len(self.matches_dataset) / matches_count:
            self.match_indices = np.random.choice(range(0, len(self.matches_dataset)),
                                                  size=(len(self.matches_dataset)), replace=True)
            self.matches_batch_index = 0

        if self.nonmatches_batch_index >= len(self.nonmatches_dataset) / nonmatches_count:
            self.nonmatch_indices = np.random.choice(range(0, len(self.nonmatches_dataset)),
                                                     size=(len(self.nonmatches_dataset)), replace=True)
            self.nonmatches_batch_index = 0

        # generate the matching pairs
        for i in range(self.matches_batch_index * matches_count, min(len(self.matches_dataset),
                                                                     (self.matches_batch_index + 1) * matches_count)):
            left_title = self.matches_dataset[self.match_indices[i]][0]
            right_title = self.matches_dataset[self.match_indices[i]][1]
            label = self.matches_dataset[self.match_indices[i]][4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])
            labels.append(label)

        # generate the non-matching pairs
        for i in range(self.nonmatches_batch_index * nonmatches_count,
                       min(len(self.matches_dataset), (self.nonmatches_batch_index + 1) * nonmatches_count)):
            left_title = self.nonmatches_dataset[self.nonmatch_indices[i]][0]
            right_title = self.nonmatches_dataset[self.nonmatch_indices[i]][1]
            label = self.nonmatches_dataset[self.nonmatch_indices[i]][4]

            left.append(self.images[left_title])
            right.append(self.images[right_title])

            labels.append(label)

        self.matches_batch_index += 1
        self.nonmatches_batch_index += 1

        left = np.array(left)
        right = np.array(right)
        labels = np.array(labels).reshape((-1, 1))
        return left, right, labels


def get_fashion_dataset(dataset_path, images_prefix):
    dataset = pickle.load(open(dataset_path, 'rb'))
    m_count = 0
    nm_count = 0
    small_dataset = []
    for d in dataset:
        if d[4] == 0 and nm_count < 300:
            small_dataset.append(d)
            nm_count += 1
        elif d[4] == 1 and m_count < 300:
            small_dataset.append(d)
            m_count += 1
        if nm_count >= 300 and m_count >= 300:
            break
    print("got small dataset")
    dataset = small_dataset
    # only keep the instances which actually have the images in the folder
    dataset = [d for d in dataset if os.path.isfile(images_prefix + d[0]) and os.path.isfile(images_prefix + d[1])]

    image_titles = list(set([d[0] for d in dataset] + [d[1] for d in dataset]))
    # divide by 255 to normalize the values
    image_list = [scim.imread(images_prefix + title) / 255.0 for title in image_titles]
    # if there are grayscale images, duplicate the channel to make it RGB
    image_list = [image if len(image.shape) == 3 else np.repeat(image[:, :, np.newaxis], 3, axis=2)
                  for image in image_list]

    image_dict = {}
    for image, image_title in zip(image_list, image_titles):
        image_dict[image_title] = image

    return dataset, image_dict


if __name__ == "__main__":
    dataset = [('group62_214351264.jpg', 'group278_212139949.jpg', 'bottom', 'top', 0),
               ('group66_208154631.jpg', 'group320_211942376.jpg', 'top', 'bottom', 0),
               ('group642_215373191.jpg', 'group162_218003027.jpg', 'top', 'bottom', 1),
               ('group292_201204397.jpg', 'group57_212453107.jpg', 'bottom', 'top', 1),
               ('group683_186868573.jpg', 'group458_192479428.jpg', 'bottom', 'top', 1)]

    dataset, image_dict = get_fashion_dataset()
    generator = BatchGenerator(dataset, image_dict)
