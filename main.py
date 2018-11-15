# -*- coding: utf-8 -*-
import sys, os
import nn
import argparse
import json
import random
import numpy as np


from PIL import Image


def compute_direction(x, y, threshold_delta):
    return 1 if x - y > threshold_delta else 0

def compute_lbp(data, i, j, threshold_delta):

    result = 0
    result += compute_direction(data[i, j], data[i-1, j], threshold_delta)
    result += 2 * compute_direction(data[i, j], data[i-1, j+1], threshold_delta)
    result += 4 * compute_direction(data[i, j], data[i, j+1], threshold_delta)
    result += 8 * compute_direction(data[i, j], data[i+1, j+1], threshold_delta)
    result += 16 * compute_direction(data[i, j], data[i+1, j], threshold_delta)
    result += 32 * compute_direction(data[i, j], data[i+1, j-1], threshold_delta)
    result += 64 * compute_direction(data[i, j], data[i, j-1], threshold_delta)
    result += 128 * compute_direction(data[i, j], data[i-1, j-1], threshold_delta)
    return result

def get_histogram(img):
    patterns = []
    pixels = list(img.getdata())
    pixels = [pixels[i * img.width:(i + 1) * img.width] for i in range(img.height)]

    # Calculate LBP for each non-edge pixel
    for i in range(1, img.height - 1):
        # Cache only the rows we need (within the neighborhood)
        previous_row = pixels[i - 1]
        current_row = pixels[i]
        next_row = pixels[i + 1]

        for j in range(1, img.width - 1):
            # Compare this pixel to its neighbors, starting at the top-left pixel and moving
            # clockwise, and use bit operations to efficiently update the feature vector
            pixel = current_row[j]
            pattern = 0
            pattern = pattern | (1 << 0) if pixel < previous_row[j - 1] else pattern
            pattern = pattern | (1 << 1) if pixel < previous_row[j] else pattern
            pattern = pattern | (1 << 2) if pixel < previous_row[j + 1] else pattern
            pattern = pattern | (1 << 3) if pixel < current_row[j + 1] else pattern
            pattern = pattern | (1 << 4) if pixel < next_row[j + 1] else pattern
            pattern = pattern | (1 << 5) if pixel < next_row[j] else pattern
            pattern = pattern | (1 << 6) if pixel < next_row[j - 1] else pattern
            pattern = pattern | (1 << 7) if pixel < current_row[j - 1] else pattern
            patterns.append(pattern)

    # img1 = Image.new(img.mode, (img.width - 2, img.height - 2))
    # img1.putdata(patterns)
    # img1.show()
    # img1.close()
    [hist, _] = np.histogram(patterns, bins=59, normed=True)
    # print hist
    return hist.tolist()

def create_training_sets(tp):
    image_paths = []
    name = []
    imgList = {'image': []}
    size = 16, 16

    dirs = os.listdir(tp)
    for dir in dirs:
        if os.path.isdir(tp + "/" + dir):
            image_paths.append(tp + "/" + dir + '/')
            name.append(dir)

    for face_idx, path in enumerate(image_paths):
        face = [0] * len(image_paths)
        face[face_idx] = 1
        imageList = filter(lambda x: x.endswith('.jpg'), os.listdir(path))
        for inImage1 in imageList:
            img = Image.open(path + inImage1)
            img = img.convert(mode='L')
            img = img.resize(size, Image.ANTIALIAS)
            histogram = get_histogram(img)
            obj = {
                'histogram': histogram,
                'face': face,
            }
            imgList['image'].append(obj)

    with open('training_set.json', 'w') as outfile:
        json.dump(imgList, outfile)

class NetworkInfo:
    num_inputs = 59
    num_hidden = 6
    num_outputs = 4

    hidden_layer_weights = None
    hidden_layer_bias = None
    output_layer_weights = None
    output_layer_bias = None

    total_error = 1

    is_read_file_error = False

    training_sets = []
    training_name = []

    def __init__(self, is_train=False, tp=''):
        if (tp != ''):
            self.num_outputs = len(os.listdir(tp))
            for dir in os.listdir(tp):
                if os.path.isdir(tp + "/" + dir):
                    self.training_name.append(dir)

        self.get_training_sets(tp=tp)
        self.get_network_from_file(is_train)

    def get_training_sets(self, tp=''):
        while True:
            try:
                with open('training_set.json') as json_data:
                    training_set = json.load(json_data)
                    for image in training_set['image']:
                        row = [image['histogram'], image['face']]
                        self.training_sets.append(row)
                break
            except (OSError, IOError):
                create_training_sets(tp)

    def get_network_from_file(self, is_train=False):
        network = None
        try:
            with open('network.json') as json_data:
                network = json.load(json_data)
        except (OSError, IOError) as error:
            if not is_train:
                print(error)
                print('if you want create new neural-network run with key -t')
                self.is_read_file_error = True
        if network:
            self.training_name = network['name_dir']
            self.num_outputs = network['output_layer']['count_neurons']
            self.num_hidden = network['hidden_layer']['count_neurons']
            self.hidden_layer_bias = []
            self.hidden_layer_weights = []
            for neuron in network['hidden_layer']['neurons']:
                self.hidden_layer_bias.append(neuron['bias'])
                for weight in neuron['weights']:
                    self.hidden_layer_weights.append(weight)

            self.output_layer_bias = []
            self.output_layer_weights = []
            for neuron in network['output_layer']['neurons']:
                self.output_layer_bias.append(neuron['bias'])
                for weight in neuron['weights']:
                    self.output_layer_weights.append(weight)
            self.total_error = network['total_error']


def train(epsilon=0.001, test_path=''):

    network = NetworkInfo(is_train=True, tp=test_path)
    NN = nn.NN(
        num_inputs=network.num_inputs,
        num_hidden=network.num_hidden,
        num_outputs=network.num_outputs,
        hidden_layer_weights=network.hidden_layer_weights,
        hidden_layer_bias=network.hidden_layer_bias,
        output_layer_weights=network.output_layer_weights,
        output_layer_bias=network.output_layer_bias,
        name_path=network.training_name,
    )
    total_error = network.total_error
    count = 0
    while float(total_error) > float(epsilon):
        try:
            training_inputs, training_outputs = random.choice(network.training_sets)
            NN.train(training_inputs, training_outputs)
            outputs = NN.feed_forward(training_inputs)
            for i in range(len(outputs)):
                outputs[i] = round(outputs[i])

            if count == 1000:
                print(outputs, training_outputs)
                total_error = NN.calculate_total_error(network.training_sets)
                print('error = ', total_error)
                network_data = NN.inspect(network.training_sets)
                with open('network.json', 'w') as outfile:
                    json.dump(network_data, outfile)
                count = 0
            else:
                count += 1
        except Exception as e:
            print(e)
            network_data = NN.inspect(network.training_sets)
            with open('network.json', 'w') as outfile:
                json.dump(network_data, outfile)

    network_data = NN.inspect(network.training_sets)
    with open('network.json', 'w') as outfile:
        json.dump(network_data, outfile)
    print(json.dumps(network_data, sort_keys=True, indent=4, separators=(',', ': ')))
    print(NN.feed_forward(network.training_sets[0][0]))
    print('Total error: ', total_error)


def main(filepath=''):
    size = 16, 16

    network = NetworkInfo()
    if network.is_read_file_error:
        return
    NN = nn.NN(
        num_inputs=network.num_inputs,
        num_hidden=network.num_hidden,
        num_outputs=network.num_outputs,
        hidden_layer_weights=network.hidden_layer_weights,
        hidden_layer_bias=network.hidden_layer_bias,
        output_layer_weights=network.output_layer_weights,
        output_layer_bias=network.output_layer_bias,
        name_path=network.training_name,
    )

    img = Image.open(filepath)
    img.show()
    img = img.convert(mode='L')
    img = img.resize(size)
    histogram = get_histogram(img)

    network_outputs = NN.feed_forward(histogram)

    print('result:')
    answer = []
    for indx, output in enumerate(network_outputs):
        print(NN.name_path[indx].title() + ':\t' + str(output))
        if round(output) == 1:
            answer.append(indx)
    if len(answer) == 0:
        index = maxToIndex(network_outputs)
        print(u'Похож на объект под имененм: ' + str(NN.name_path[index]))
    elif len(answer) > 1:
        index = maxOfOutputsToIndex(answer, network_outputs)
        print(u'Скорее всего это объект: ' + str(NN.name_path[index]))
    else:
        print(u'Объект: ' + str(NN.name_path[answer[0]]))


def maxToIndex(vList):
    max = 0
    indexOfMax = 0
    for indx, output in enumerate(vList):
        if output > max:
            max = output
            indexOfMax = indx
    return indexOfMax


def maxOfOutputsToIndex(answer, network_outputs):
    max = 0
    indexOfMax = 0
    for output in answer:
        if network_outputs[output] > max:
            max = network_outputs[output]
            indexOfMax = output
    return indexOfMax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face recognition using neural network.')
    parser.add_argument('-t', '--train', type=str, default=None, help="train neural-network or create new (standart error = 0.0001)")
    parser.add_argument('-dt', '--dir_train', type=str, default=None, help='test folder')
    parser.add_argument('-f', '--file', type=str, default=None, help="get image season")
    options = parser.parse_args()
    if options.file:
        try:
            main(options.file)
        except IndexError as e:
            print (e)
            sys.exit()

    if options.train:
        try:
            train(options.train, options.dir_train)
        except IndexError as e:
            print (e)
            sys.exit()