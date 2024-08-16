import argparse
import itertools
import math
import numpy as np
import sys
import os

from glob import glob
from data import Data, AVAILABLE_VIEWPOINTS, NOT_USELESS_VIEWPOINTS, DELIMITER, CPITCH_VP, get_representations
from idyom import Idyom
from tqdm import tqdm

from feature_selection import manzara_feature_selection, uncertainty_feature_selection, cross_validation

from variance_comparison import model_comparison, MANZARA_IC_151, MANZARA_IC_61
reference_data = MANZARA_IC_151

import matplotlib.pyplot as plt


MODEL_DIR = "models"

def get_files_from_path(path):
    files = []
    for filename in glob(path + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            files.append(filename)
    return files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--viewpoints", default=NOT_USELESS_VIEWPOINTS, help="viewpoints to use")
    parser.add_argument("-t", "--train", help="train the model on this path")
    parser.add_argument("-s", "--surprise", default=None, help="get the surprise for scores in this path")
    parser.add_argument("-q", "--quantization", default=24, type=int, help="number of ticks in a quarter note")
    parser.add_argument("-w", "--write", default="output.txt", type=str, help="Output file")
    parser.add_argument("--enable_augmentation", action="store_false" ,help="augment the data or not")
    parser.add_argument("--long_term_only", action="store_true" ,help="Use only the long term model (LTM)")
    parser.add_argument("-c", "--cross_validation", help="Perform cross_validation on this path")
    parser.add_argument("-k", "--folds", default=10, type=int, help="Number of folds for cross validation")
    parser.add_argument("-f", "--manzara_feature_selection", help="Perform feature selection to fit behavioral data from Manzara")
    parser.add_argument("-u", "--uncertainty_feature_selection", help="Perform feature selection reduce the uncertainty")
    args = parser.parse_args()

    if not isinstance(args.viewpoints, list):
        args.viewpoints = args.viewpoints.split(',')

    for elem in args.viewpoints:
        if DELIMITER in elem:
            vps = [vp for vp in elem.split(DELIMITER)]
        else:
            vps = [elem]
        for vp in vps:
            if vp not in AVAILABLE_VIEWPOINTS:
                print(f"viewpoint {vp} is not available. Check your spelling")
                sys.exit(1)

    augmentation = False if args.enable_augmentation else True
    use_stm = False if args.long_term_only else True

    if args.train:
        if not os.path.exists(args.train):
            print(f"path {args.train} does not exist")
            sys.exit(1)

        vps = ""
        for vp, i in zip(args.viewpoints, range(len(args.viewpoints))):
            vps += vp
            if i != len(args.viewpoints):
                vps += "_"

        model_name = f'{MODEL_DIR}/{args.train.replace("/", "_")}_quantization_{args.quantization}_augmentation_{augmentation}_{vps}.model'

        files = get_files_from_path(args.train)
        data = Data(files, args.quantization)
        idyom = Idyom(args.viewpoints, args.quantization)
        train = True

        if os.path.exists(model_name):
            train = True if input("Model already exists, retrain ? (y/n): ").strip().lower() == 'y' else False

        if train:
            data.parse(augment=augmentation)
            representation, pitch_representation = get_representations(data, files, args.viewpoints)
            print(f"\n____ Training ____\n")
            idyom.train(representation, pitch_representation)

            print(f"\n____ Saving {model_name} ____")
            idyom.save(model_name)

        else:
            idyom.load(model_name)

        if args.surprise:
            if not os.path.exists(args.surprise):
                print(f"path {args.surprise} does not exist")
                sys.exit(1)
            else:
                files = get_files_from_path(args.surprise)
                test_data = Data(files, args.quantization)
                test_data.parse(augment=False) # Do note augment the test data

                representation, pitch_representation = get_representations(test_data, files, args.viewpoints)

                f = open(args.write, "a")
                count = len(test_data.scores)
                for num_score, name in zip(range(count), test_data.scores):
                    sequence = {}
                    for vp in idyom.viewpoints:
                        sequence[vp] = representation[vp][num_score]
                    pitch_sequence = pitch_representation[num_score]
                    probas = idyom.get_likelihood_from_sequence(sequence, pitch_sequence, use_stm)
                    ics = idyom.get_information_content(probas)
                    print(name)
                    print(ics)
                    f.write(name + " : ")
                    f.write(' '.join([str(e) for e in ics]))
                    f.write('\n')
                f.close()



    if args.cross_validation:

        if not os.path.exists(args.cross_validation):
            print(f"path {args.cross_validation} does not exist")
            sys.exit(1)
        files = get_files_from_path(args.cross_validation)
        np.random.shuffle(files)
        all_data = Data(files, args.quantization)
        all_data.parse(augment=augmentation)
        mean_cross_entropy = cross_validation(all_data, args.viewpoints, args.folds, files, use_stm)
        print('mean cross_entropy = ', mean_cross_entropy)

    if args.manzara_feature_selection:
        if not os.path.exists(args.manzara_feature_selection):
            print(f"path {args.manzara_feature_selection} does not exist")
            sys.exit(1)
        files = get_files_from_path(args.manzara_feature_selection)
        np.random.shuffle(files)
        all_data = Data(files, args.quantization)
        all_data.parse(augment=augmentation)

        if not os.path.exists(args.surprise):
            print(f"path {args.surprise} does not exist")
            sys.exit(1)
        test_files = get_files_from_path(args.surprise)
        test_data = Data(test_files, args.quantization)
        test_data.parse(augment=False)
        to_compare = {'dataset/manzara/chor151.mid' : MANZARA_IC_151,
                      'dataset/manzara/chor061.mid' : MANZARA_IC_61}
        manzara_feature_selection(all_data, test_data, files, test_files, args.viewpoints, use_stm, to_compare)

    if args.uncertainty_feature_selection:
        if not os.path.exists(args.uncertainty_feature_selection):
            print(f"path {args.uncertainty_feature_selection} does not exist")
            sys.exit(1)
        files = get_files_from_path(args.uncertainty_feature_selection)
        np.random.shuffle(files)
        all_data = Data(files, args.quantization)
        all_data.parse(augment=augmentation)
        uncertainty_feature_selection(all_data, files, args.viewpoints, use_stm, args.folds)
