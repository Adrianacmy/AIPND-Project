import argparse
from time import time
import json
import torch
import helper


def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='Input image')
    parser.add_argument('checkpoint', type=str,
                        help='Model checkpoint file to use for prediction')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top k most likely classes')
    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU for prediction')
    parser.add_argument('--category_names', type=str,
                        help='Mapping file used to map categories to real names')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():

    input_args = get_input_args()
    gpu = torch.cuda.is_available() and input_args.gpu
    print("Predicting on {} using {}".format(
        "GPU" if gpu else "CPU", input_args.checkpoint))

    model = helper.load_checkpoint(input_args.checkpoint)

    if gpu:
        model.cuda()

    use_mapping_file = False

    if input_args.category_names:
        with open(input_args.category_names, 'r') as f:
            cat_to_name = json.load(f)
            use_mapping_file = True

    probs, classes = helper.predict(input_args.input, model, gpu, input_args.top_k)

    print("Top {} classes for '{}' :".format(len(classes), input_args.input))

    for i in range(input_args.top_k):
        print("probability of class {}: {}".format(classes[i], probs[i]))


if __name__ == "__main__":
    main()
