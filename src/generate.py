import argparse
import numpy as np
import os
import subprocess
import sys
from music21 import stream, midi, note, duration, meter, tempo
import os
from glob import glob
from scipy.stats import linregress

from data import Data, AVAILABLE_VIEWPOINTS, CPITCH_VP, DUR_VP, DELIMITER
from idyom import Idyom
from tqdm import tqdm
from tex import from_experiment_to_tex
from evaluation import load_evaluation_model, evaluate, DATASET_PATH, REGRESSION_PATH

import matplotlib.pyplot as plt

from copy import deepcopy

EXPERIMENT_DIR = "experiments/"

from datetime import datetime

def get_files_from_path(path):
    files = []
    for filename in glob(path + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            files.append(filename)
    return files

def get_experiment_folder():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_folder = f"experiment_{timestamp}/"
    return experiment_folder

class Generator:
    def __init__(self, base_score_path, length, output_file, model_name):
        self.idyom = Idyom()
        self.idyom.load(model_name)
        self.quantization_factor = self.idyom.quantization
        self.model_name = model_name

        self.base_score_path = base_score_path
        self.length = length
        self.output_file = output_file

    def getPairs(self):
        alphabets = []
        for vp_model in self.idyom.models.values():
            vp_alphabet = vp_model.getAlphabet()
            alphabets.append(vp_alphabet)
        pairs = self.generate_variables(alphabets)
        return pairs

    def generate_variables(self, lists):
        if len(lists) == 0:
            return [[]]
        else:
            pairs = []
            for item in lists[0]:
                for pair in self.generate_variables(lists[1:]):
                    pairs.append([item] + pair)
            return pairs

    def generate(self, temperature=1.0):
        if self.base_score_path != None:
            files = [self.base_score_path]
            data = Data(files, self.quantization_factor)
            data.parse(augment=False) # Do note augment the test data
            size = data.scores[self.base_score_path][0].size()
            durations = data.scores[self.base_score_path][0].duration
        else:
            durations = None
            size = self.length

        S = []
        probas = []
        base_score = data.scores[self.base_score_path][0]
        # pairs = self.getPairs()
        for i in tqdm(range(size)):

            current_score = deepcopy(base_score)
            # Trim
            current_score.pitch = S
            current_score.duration = base_score.duration[0:i]
            current_score.onset = base_score.onset[0:i]
            current_score.velocity = base_score.velocity[0:i]

            note, prob = self.idyom.sample(current_score, temperature)
            probas.append(prob)
            S.append(note)
        sequence_cross_entropy = self.idyom.get_cross_entropy(probas)
        return S, sequence_cross_entropy, durations

    def output_midi(self, pitches, dur, output_file):

        durations = [e/self.quantization_factor for e in dur]

        melody = stream.Stream()

        for midi_note, dur in zip(pitches, durations):
            note_obj = note.Note()
            note_obj.pitch.midi = midi_note
            note_obj.duration = duration.Duration(dur)
            melody.append(note_obj)

        melody.insert(0, tempo.MetronomeMark(number=100))
        melody.write('midi', output_file)
        # melody.show('midi')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="The model to use")
    parser.add_argument("-o", "--output", default='output.mid', help="name of the output midi file")
    parser.add_argument("-l", "--length", type=int, default=16, help="length of the melody generated")
    parser.add_argument("-i", "--initial_state", help="The initial state for metropolis sampling")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="")

    args = parser.parse_args()

    if args.model:
        if not os.path.exists(args.model):
            print(f"path {args.model} does not exist")
            sys.exit(1)

    model_name = args.model

    generator = Generator(args.initial_state, args.length, args.output, model_name)
    # generated_sequence, sequence_cross_entropy, durations = generator.generate(args.temperature)
    # generator.output_midi(generated_sequence, durations, args.output)
    # model = load_evaluation_model(DATASET_PATH, REGRESSION_PATH)
    # rating = evaluate(model, args.output, DATASET_PATH)
    # print(rating)

    model = load_evaluation_model(DATASET_PATH, REGRESSION_PATH)
    ratings = []

    # Step 1: Fetch the base melodies
    base_melodies = get_files_from_path("./dataset/base_chorales/")

    # Step 2: Define the temperatures
    temperatures = [0.02, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    mean_ratings = []
    all_ratings = []

    # Step 3: Iterate over each temperature and base melody to generate and evaluate sequences
    k = 0
    for temp in temperatures:
        temp_ratings = []
        for base_melody in base_melodies:
            generator.base_score_path = base_melody
            for _ in range(10):
                k += 1
                print(k)
                durations = None
                generated_sequence, sequence_cross_entropy, durations = generator.generate(temp)
                generator.output_midi(generated_sequence, durations, args.output)
                rating = evaluate(model, args.output, DATASET_PATH)
                temp_ratings.append(rating)
        mean_rating = np.mean(temp_ratings)
        mean_ratings.append(mean_rating)
        all_ratings.append(temp_ratings)


    mean_ratings = np.array(mean_ratings)
    std_devs = np.array([np.std(ratings) for ratings in all_ratings])

    print()
    print()
    print()

    with open("output.txt", "a") as file:
        for t, m, s in zip(temperatures, mean_ratings, std_devs):
            file.write(f"({t},{m}) +- (0, {s})\n")
        
        file.write("\n")
