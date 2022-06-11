import glob
import os
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import consts
from enum import Enum


class Class(Enum):
    CANTALOUPE = 0
    BANANA = 1
    TOMATO = 2


class BlockType(Enum):
    SLIDING = True
    BLOCK = False


class MergeType(Enum):
    RANDOM = True
    NON_OVERLAPPING = False


def main():

    # Get all images from the path
    banana_image_files = get_image_files(consts.BANANA_IMAGES_PATH)
    cantaloupe_image_files = get_image_files(consts.CANTALOUPE_IMAGES_PATH)
    tomato_image_files = get_image_files(consts.TOMATO_IMAGES_PATH)

    # Read the image
    banana_color_image = read_image(banana_image_files[33])
    cantaloupe_color_image = read_image(cantaloupe_image_files[17])
    tomato_color_image = read_image(tomato_image_files[43])

    # Display the BGR Channel
    show_BGR_image(banana_color_image)
    show_BGR_image(cantaloupe_color_image)
    show_BGR_image(tomato_color_image)

    # Display the dimensions of the first image
    display_image_dimensions(banana_color_image)
    display_image_dimensions(cantaloupe_color_image)
    display_image_dimensions(tomato_color_image)

    # Convert the image to grayscale
    banana_gray_image = convert_to_grayscale(banana_color_image)
    cantaloupe_gray_image = convert_to_grayscale(cantaloupe_color_image)
    tomato_gray_image = convert_to_grayscale(tomato_color_image)

    # Resize the image
    banana_resized_image = resize(banana_gray_image)
    cantaloupe_resized_image = resize(cantaloupe_gray_image)
    tomato_resized_image = resize(tomato_gray_image)

    # Normalize the image
    banana_normalized_image = normalize_image(banana_resized_image)
    cantaloupe_normalized_image = normalize_image(cantaloupe_resized_image)
    tomato_normalized_image = normalize_image(tomato_resized_image)

    # Display the dimensions of the grayscale image
    display_image_dimensions(banana_normalized_image)
    display_image_dimensions(cantaloupe_normalized_image)
    display_image_dimensions(tomato_normalized_image)

    # Show the grayscale image
    show_grayscale_image(banana_resized_image)
    show_grayscale_image(cantaloupe_resized_image)
    show_grayscale_image(tomato_resized_image)

    # Store the image
    store_resized_grayscale_image(banana_image_files[33], banana_resized_image)
    store_resized_grayscale_image(cantaloupe_image_files[17], cantaloupe_resized_image)
    store_resized_grayscale_image(tomato_image_files[43], tomato_resized_image)

    # Display the statistics of the grayscale image
    display_image_statistics(banana_resized_image)
    display_image_statistics(cantaloupe_resized_image)
    display_image_statistics(tomato_resized_image)

    # Binarize the image
    banana_binary_image = binarize_image(banana_normalized_image)
    cantaloupe_binary_image = binarize_image(cantaloupe_normalized_image)
    tomato_binary_image = binarize_image(tomato_normalized_image)

    # Display the binary image
    show_binary_image(banana_binary_image)
    show_binary_image(cantaloupe_binary_image)
    show_binary_image(tomato_binary_image)

    # Generate sliding block-feature vectors
    banana_sliding_block_feature_vectors = generate_sliding_block_feature_vectors(banana_resized_image, Class.BANANA)
    cantaloupe_sliding_block_feature_vectors = generate_sliding_block_feature_vectors(cantaloupe_resized_image,
                                                                                      Class.CANTALOUPE)
    tomato_sliding_block_feature_vectors = generate_sliding_block_feature_vectors(tomato_resized_image, Class.TOMATO)

    # Store the sliding block-feature space
    store_sliding_block_feature_space(banana_sliding_block_feature_vectors, banana_image_files[33])
    store_sliding_block_feature_space(cantaloupe_sliding_block_feature_vectors, cantaloupe_image_files[17])
    store_sliding_block_feature_space(tomato_sliding_block_feature_vectors, tomato_image_files[43])

    # Generate block-feature vectors
    banana_block_feature_vectors = generate_block_feature_vectors(banana_resized_image, Class.BANANA)
    cantaloupe_block_feature_vectors = generate_block_feature_vectors(cantaloupe_resized_image, Class.CANTALOUPE)
    tomato_block_feature_vectors = generate_block_feature_vectors(tomato_resized_image, Class.TOMATO)

    # Store the block-feature space
    store_block_feature_space(banana_block_feature_vectors, banana_image_files[33])
    store_block_feature_space(cantaloupe_block_feature_vectors, cantaloupe_image_files[17])
    store_block_feature_space(tomato_block_feature_vectors, tomato_image_files[43])

    # Display statistical description of the feature space
    print("Banana Block Feature Space")
    display_feature_space_statistics(banana_block_feature_vectors)
    print("Cantaloupe Block Feature Space")
    display_feature_space_statistics(cantaloupe_block_feature_vectors)
    print("Tomato Block Feature Space")
    display_feature_space_statistics(tomato_block_feature_vectors)
    print("Banana Sliding Block Feature Space")
    display_feature_space_statistics(banana_sliding_block_feature_vectors)
    print("Cantaloupe Sliding Block Feature Space")
    display_feature_space_statistics(cantaloupe_sliding_block_feature_vectors)
    print("Tomato Sliding Block Feature Space")
    display_feature_space_statistics(tomato_sliding_block_feature_vectors)

    # Display statistical graphs of the feature space
    banana_sliding_block_mean = banana_sliding_block_feature_vectors.iloc[:, 0:80].mean()
    cantaloupe_sliding_block_mean = cantaloupe_sliding_block_feature_vectors.iloc[:, 0:80].mean()
    tomato_sliding_block_mean = tomato_sliding_block_feature_vectors.iloc[:, 0:80].mean()
    banana_block_mean = banana_block_feature_vectors.iloc[:, 0:80].mean()
    cantaloupe_block_mean = cantaloupe_block_feature_vectors.iloc[:, 0:80].mean()
    tomato_block_mean = tomato_block_feature_vectors.iloc[:, 0:80].mean()

    banana_sliding_block_std = banana_sliding_block_feature_vectors.iloc[:, 0:80].std()
    cantaloupe_sliding_block_std = cantaloupe_sliding_block_feature_vectors.iloc[:, 0:80].std()
    tomato_sliding_block_std = tomato_sliding_block_feature_vectors.iloc[:, 0:80].std()
    banana_block_std = banana_block_feature_vectors.iloc[:, 0:80].std()
    cantaloupe_block_std = cantaloupe_block_feature_vectors.iloc[:, 0:80].std()
    tomato_block_std = tomato_block_feature_vectors.iloc[:, 0:80].std()

    show_feature_statistics_plot(banana_block_mean, Class.BANANA, cantaloupe_block_mean, Class.CANTALOUPE, tomato_block_mean,
                            Class.TOMATO)
    show_feature_statistics_plot(banana_sliding_block_mean, Class.BANANA, cantaloupe_sliding_block_mean, Class.CANTALOUPE,
                            tomato_sliding_block_mean, Class.TOMATO)

    show_feature_statistics_plot(banana_block_std, Class.BANANA, cantaloupe_block_std, Class.CANTALOUPE, tomato_block_std,
                            Class.TOMATO)
    show_feature_statistics_plot(banana_sliding_block_std, Class.BANANA, cantaloupe_sliding_block_std, Class.CANTALOUPE,
                            tomato_sliding_block_std, Class.TOMATO)

    # Merge the block-feature vectors
    merged_two_class_block_feature_vectors = merge_feature_vectors(banana_block_feature_vectors,
                                                                   cantaloupe_block_feature_vectors)
    merged_three_class_block_feature_vectors = merge_feature_vectors(banana_block_feature_vectors,
                                                                     cantaloupe_block_feature_vectors,
                                                                     tomato_block_feature_vectors)
    merged_two_class_sliding_block_feature_vectors = merge_feature_vectors(banana_sliding_block_feature_vectors,
                                                                           cantaloupe_sliding_block_feature_vectors)
    merged_three_class_sliding_block_feature_vectors = merge_feature_vectors(banana_sliding_block_feature_vectors,
                                                                             cantaloupe_sliding_block_feature_vectors,
                                                                             tomato_sliding_block_feature_vectors)
    random_merged_two_class_block_feature_vectors = random_merge_feature_vectors(banana_block_feature_vectors,
                                                                                 cantaloupe_block_feature_vectors)
    random_merged_three_class_block_feature_vectors = random_merge_feature_vectors(banana_block_feature_vectors,
                                                                                   cantaloupe_block_feature_vectors,
                                                                                   tomato_block_feature_vectors)
    random_merged_two_class_sliding_block_feature_vectors = random_merge_feature_vectors(
        banana_sliding_block_feature_vectors, cantaloupe_sliding_block_feature_vectors)
    random_merged_three_class_sliding_block_feature_vectors = random_merge_feature_vectors(
        banana_sliding_block_feature_vectors, cantaloupe_sliding_block_feature_vectors,
        tomato_sliding_block_feature_vectors)

    # Store the merged block-feature space
    store_merged_feature_space(merged_two_class_block_feature_vectors, Class.BANANA, Class.CANTALOUPE)
    store_merged_feature_space(merged_three_class_block_feature_vectors, Class.BANANA, Class.CANTALOUPE, Class.TOMATO)
    store_merged_feature_space(merged_two_class_sliding_block_feature_vectors, Class.BANANA, Class.CANTALOUPE, None,
                               True)
    store_merged_feature_space(merged_three_class_sliding_block_feature_vectors, Class.BANANA, Class.CANTALOUPE,
                               Class.TOMATO, True)
    store_merged_feature_space(random_merged_two_class_block_feature_vectors, Class.BANANA, Class.CANTALOUPE, None,
                               False, True)
    store_merged_feature_space(random_merged_three_class_block_feature_vectors, Class.BANANA, Class.CANTALOUPE,
                               Class.TOMATO, False, True)
    store_merged_feature_space(random_merged_two_class_sliding_block_feature_vectors, Class.BANANA, Class.CANTALOUPE,
                               None, True, True)
    store_merged_feature_space(random_merged_three_class_sliding_block_feature_vectors, Class.BANANA, Class.CANTALOUPE,
                               Class.TOMATO, True, True)

    # Display plots of select features for each class
    bana_cant_csv = read_csv(consts.DATA_PATH + r'\10_merged_block_feature_space.csv')
    bana_cant_sliding_csv = read_csv(consts.DATA_PATH + r'\10_merged_sliding_block_feature_space.csv')
    bana_cant_toma_csv = read_csv(consts.DATA_PATH + r'\102_merged_block_feature_space.csv')
    bana_cant_toma_sliding_csv = read_csv(consts.DATA_PATH + r'\102_merged_sliding_block_feature_space.csv')

    show_select_fetures_plot(bana_cant_csv)
    show_select_fetures_plot(bana_cant_toma_csv, 3)
    show_select_fetures_plot(bana_cant_sliding_csv)
    show_select_fetures_plot(bana_cant_toma_sliding_csv, 3)


def get_image_files(path):
    """
    Get all images from the path
    :param path: path to the images
    :return: list of image files
    """


    images = glob.glob(path + r'\\*.jpg')
    return images


def read_image(file):
    """
    Read the image file
    :param file: path to the image
    :return: BGR image
    """

    image_color = cv.imread(file)
    return image_color


def show_BGR_image(image):
    """
    Display the BGR image
    :param image: image
    """

    plt.imshow(image[:, :, 0])  # Blue
    plt.imshow(image[:, :, 1])  # Green
    plt.imshow(image[:, :, 2])  # Red
    plt.show()


def display_image_dimensions(image):
    """
    Display the dimensions of an image
    :param image: image
    """

    height, width = image.shape[:2]
    print("Image dimensions: {}".format((height, width)))

# https://pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/
def resize(image):
    """
    Resize the image
    :param image: image
    :return: resized image
    """

    height, width = image.shape

    ratio = 261 / height

    new_width = int(width * ratio)
    new_width_offset = new_width % 9
    new_width = new_width - new_width_offset

    # new dimensions
    dim = (new_width, 261)

    return cv.resize(image, dim, interpolation=cv.INTER_AREA)


def convert_to_grayscale(image):
    """
    Display the grayscale image
    :param image: image
    :return: grayscale image
    """

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray


def show_grayscale_image(image):
    """
    Display the grayscale image
    :param image: image
    """

    plt.imshow(image, cmap='gray')
    plt.show()


def normalize_image(image):
    """
    Normalize the image
    :param image: image
    :return: normalized image
    """

    image_normalized = cv.normalize(image.astype('float'), None, 0.0, 1.0, norm_type=cv.NORM_MINMAX) * 255
    return image_normalized


def store_resized_grayscale_image(path, image):
    """
    Store the image
    :param image: image
    :param path: path to the image
    """

    filename = r'\\' + os.path.splitext(os.path.basename(path))[0] + '_resized.jpg'
    file = consts.DATA_PATH + filename
    cv.imwrite(file, image)


def display_image_statistics(image):
    """
    Display the statistics of the image
    :param image: image
    """

    print("Statistics:")
    print("Min: {}".format(image.min()))
    print("Max: {}".format(image.max()))
    print("Mean: {}".format(round(image.mean(), 3)))
    print("Standard deviation: {}".format(round(image.std(), 4)))


def binarize_image(image):
    """
    Binarize the image
    :param image: image
    :return: binarized image
    """

    height, width = image.shape
    image_binarized = np.zeros((height, width), np.uint8)
    threshold = image.mean()
    for i in range(height):
        for j in range(width):
            if image[i, j] < threshold:
                image_binarized[i, j] = 0
            else:
                image_binarized[i, j] = 255

    return image_binarized


def show_binary_image(image):
    """
    Display the binarized image
    :param image: image
    """

    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


def generate_sliding_block_feature_vectors(image, label):
    """
    Generate the sliding block feature vectors
    :param label: label
    :param image: image
    :return: feature vectors
    """

    height, width = image.shape

    l = round((height * width))
    if label == Class.CANTALOUPE:
        flat = np.zeros((l, 82), np.uint8)
    elif label == Class.BANANA:
        flat = np.ones((l, 82), np.uint8)
    elif label == Class.TOMATO:
        flat = np.ones((l, 82), np.uint8)
        flat[:] = [i + 1 for i in flat]
    else:
        print("Error: Invalid label")
        return

    k = 0
    for i in range(height - 8):
        for j in range(width - 8):
            crop_tmp = image[i:i + 9, j:j + 9]
            flat[k, 0:81] = crop_tmp.flatten()
            k = k + 1

    feature_space = pd.DataFrame(flat)
    return feature_space


def store_sliding_block_feature_space(feature_space, original_file):
    """
    Store the feature space
    :param feature_space: feature space
    :param original_file: file
    """

    filename = r'\\' + os.path.splitext(os.path.basename(original_file))[0] + '_sliding_block_feature_space.csv'
    file = consts.DATA_PATH + filename
    feature_space.to_csv(file, index=False)


def generate_block_feature_vectors(image, label):
    """
    Generate the block feature vectors
    :param image:
    :param label:
    :return:
    """

    height, width = image.shape

    l = round((height * width) / 81)
    if label == Class.CANTALOUPE:
        flat = np.zeros((l, 82), np.uint8)
    elif label == Class.BANANA:
        flat = np.ones((l, 82), np.uint8)
    elif label == Class.TOMATO:
        flat = np.ones((l, 82), np.uint8)
        flat[:] = [i + 1 for i in flat]
    else:
        print("Error: Invalid label")
        return
    k = 0
    for i in range(0, height, 9):
        for j in range(0, width, 9):
            crop_tmp = image[i:i + 9, j:j + 9]
            flat[k, 0:81] = crop_tmp.flatten()
            k = k + 1

    feature_space = pd.DataFrame(flat)
    return feature_space


def store_block_feature_space(feature_space, original_file):
    """
    Store the feature space
    :param feature_space: feature space
    :param original_file: file
    """

    filename = r'\\' + os.path.splitext(os.path.basename(original_file))[0] + '_block_feature_space.csv'
    file = consts.DATA_PATH + filename
    feature_space.to_csv(file, index=False)


def merge_feature_vectors(feature_space_1, feature_space_2, feature_space_3=None):
    """
    Merge the feature vectors
    :param feature_space_1: feature space 1
    :param feature_space_2: feature space 2
    :param feature_space_3: feature space 3
    :return: merged feature space
    """

    if feature_space_3 is not None:
        frames = [feature_space_1, feature_space_2, feature_space_3]
    else:
        frames = [feature_space_1, feature_space_2]
    merged_feature_space = pd.concat(frames)
    return merged_feature_space


def random_merge_feature_vectors(feature_space_1, feature_space_2, feature_space_3=None):
    """
    Randomly merge the feature vectors
    :param feature_space_1: feature space 1
    :param feature_space_2: feature space 2
    :param feature_space_3: feature space 3
    :return: merged feature space
    """

    if feature_space_3 is not None:
        frames = [feature_space_1, feature_space_2, feature_space_3]
    else:
        frames = [feature_space_1, feature_space_2]
    merged_feature_space = pd.concat(frames)
    index = np.arange(len(merged_feature_space))
    randomly_merged_feature_space = np.random.permutation(index)
    randomly_merged_feature_space = merged_feature_space.sample(frac=1).reset_index(drop=True)
    return randomly_merged_feature_space


def store_merged_feature_space(feature_space, label_1, label_2, label_3=None, sliding=False, random=False):
    """
    Store the feature space
    :param feature_space: feature space
    :param label_1: label 1
    :param label_2: label 2
    :param label_3: label 3
    :param sliding: sliding block feature space or not
    :param random: randomly merged or not
    """

    if label_3 is not None:
        if sliding is not False:
            if random is not False:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + str(
                    label_3.value) + '_random_merged_sliding_block_feature_space.csv'
            else:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + str(
                    label_3.value) + '_merged_sliding_block_feature_space.csv'
        else:
            if random is not False:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + str(
                    label_3.value) + '_random_merged_block_feature_space.csv'
            else:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + str(
                    label_3.value) + '_merged_block_feature_space.csv'
    else:
        if sliding is not False:
            if random is not False:
                filename = r'\\' + str(label_1.value) + str(
                    label_2.value) + '_random_merged_sliding_block_feature_space.csv'
            else:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + '_merged_sliding_block_feature_space.csv'
        else:
            if random is not False:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + '_random_merged_block_feature_space.csv'
            else:
                filename = r'\\' + str(label_1.value) + str(label_2.value) + '_merged_block_feature_space.csv'

    file = consts.DATA_PATH + filename
    feature_space.to_csv(file, index=False)


def display_feature_space_statistics(feature_space):
    """
    Display the statistics of the feature space
    :param feature_space: feature space
    :return:
    """

    print("# Observations:{}".format(feature_space.shape[0]))
    # print("Min: {}".format(feature_space.min()))
    # print("Max: {}".format(feature_space.max()))
    print("Mean: {}".format(feature_space.mean()))
    # print("Standard Deviation: {}".format(feature_space.std()))
    # print("Variance: {}".format(feature_space.var()))


def show_feature_statistics_plot(feature_space_means_1, label_1, feature_space_means_2, label_2, feature_space_means_3,
                            label_3):
    """
    Show the plot of the feature means
    :param feature_space_means_1: feature space means 1
    :param feature_space_means_2: feature space means 2
    :param feature_space_means_3: feature space means 3
    :param label_1: label 1
    :param label_2: label 2
    :param label_3: label 3
    :return:
    """

    plt.plot(feature_space_means_1, label=label_1.name, color='gold')
    plt.plot(feature_space_means_2, label=label_2.name, color='green')
    plt.plot(feature_space_means_3, label=label_3.name, color='red')
    plt.legend()
    plt.show()


def read_csv(filename):
    """
    Read the csv file
    :param filename: csv file name
    :return: numpy array of the csv file
    """
    csv = pd.read_csv(filename, header=None)
    nparray = np.array(csv)
    return nparray


def show_select_fetures_plot(merged_feature_space, num_features=2):
    """
    Show the plot of select features
    :param merged_feature_space: merged feature space
    :param num_features: number of features
    :return:
    """
    NN = merged_feature_space.shape[0]
    labels = merged_feature_space[1:NN, 81]

    if num_features == 3:
        colors = ['gold', 'green', 'red']
        ax = plt.axes(projection='3d')
        ax.scatter3D(merged_feature_space[1:NN, 11], merged_feature_space[1:NN, 32], merged_feature_space[1:NN, 59],
                     c=labels, cmap=matplotlib.colors.ListedColormap(colors))
        ax.set_xlabel('Feature 11')
        ax.set_ylabel('Feature 32')
        ax.set_zlabel('Feature 59')
    elif num_features == 2:
        colors = ['gold', 'green']
        plt.scatter(merged_feature_space[1:NN, 11], merged_feature_space[1:NN, 32], c=labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel('Feature 11')
        plt.ylabel('Feature 32')
    else:
        print("Invalid number of features")
        return
    plt.show()


if __name__ == '__main__':
    main()
