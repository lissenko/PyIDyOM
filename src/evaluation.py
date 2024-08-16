import numpy as np
import sys
from music21 import converter, note, scale, pitch
from glob import glob
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm

FEATURES = ['pitch_centre', 'pitch_range', 'interval_size',
            'interval_dissonance', 'chromaticism', 'harmonic_closure']

RATINGS = {'249' : 6.44, '238' : 5.31, '365' : 6.25, '264' : 6.00, '44' : 6.12,
           '141' : 5.50, '147' : 6.50, 'A249' : 2.56, 'A238' : 3.31, 'A365' :
           2.69, 'A264' : 1.75, 'A44' : 4.25, 'A141' : 3.38, 'A147' : 2.38,
           'B249' : 2.44, 'B238' : 2.94, 'B365' : 1.69, 'B264' : 2.00, 'B44' :
           4.38, 'B141' : 2.12, 'B147' : 1.88, 'C249' : 5.00, 'C238' : 3.19,
           'C365' : 2.50, 'C264' : 2.38, 'C44' : 4.00, 'C141' : 3.19, 'C147' :
           1.94}

DATASET_PATH = './dataset/bach-370-chorales-manzaraless/'
REGRESSION_PATH = './dataset/regression/'

def remove_extension(file_path):
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return filename_without_ext

def get_files_from_path(path):
    files = []
    for filename in glob(path + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            files.append(filename)
    return files

def get_pitches(path):
    midi_stream = converter.parse(path)
    score = midi_stream.flatten()
    pitches = [element.pitch.midi for element in score.notes if isinstance(element, note.Note)]
    return pitches

def get_mean_pitch_from_dataset(dataset_path):
    files = get_files_from_path(dataset_path)
    pitches = []
    for midiFile in files:
        midi_stream = converter.parse(midiFile)
        score = midi_stream.flatten()
        for element in score.notes:
            if isinstance(element, note.Note):
                pitches.append(element.pitch.midi)
    mean_midi_pitch = round(np.mean(pitches))
    return mean_midi_pitch

def get_mean_pitch_range_from_dataset(dataset_path):
    files = get_files_from_path(dataset_path)
    ranges = []
    for midiFile in files:
        pitches = get_pitches(midiFile)
        r = max(pitches) - min(pitches)
        ranges.append(r)
    mean_range = np.mean(ranges)
    return round(mean_range)

def pitch_centre(melody_path, dataset_mean_pitch):
    pitches = get_pitches(melody_path)
    mean_pitch = round(np.mean(pitches))
    return abs(dataset_mean_pitch - mean_pitch)

def pitch_range(melody_path, dataset_mean_range):
    pitches = get_pitches(melody_path)
    r = max(pitches) - min(pitches)
    return abs(dataset_mean_range - r)

def interval_size(melody_path):
    pitches = get_pitches(melody_path)
    intervals = list(np.abs(np.diff(pitches)))
    return len([e for e in intervals if e > 12])

def is_dissonant(interval):
    # Tritone, minor seventh, major seventh
    return (interval%12) in [6, 10, 11]

def interval_dissonance(melody_path):
    pitches = get_pitches(melody_path)
    intervals = list(np.abs(np.diff(pitches)))
    return len([e for e in intervals if is_dissonant(e)])

def total_diss(dataset_path):
    count = 0
    files = get_files_from_path(dataset_path)
    for midiFile in files:
        count += interval_dissonance(midiFile)
    return count

def get_key_signature(melody_path):
    midi_stream = converter.parse(melody_path)
    score = midi_stream.flatten()
    key_signature = score.getElementsByClass('KeySignature')
    if not key_signature:
        # key_signature = score.analyze('key')
        key_signature = score.analyze('key.krumhanslschmuckler')
    else:
        key_signature = key_signature[0]
    return key_signature

def chromaticism(melody_path):
    pitches = get_pitches(melody_path)
    key_signature = get_key_signature(melody_path)
    scale_name = key_signature.getScale()
    scale_pitches = np.array([e.midi for e in scale_name.getPitches()]) % 12
    pitches = np.array(pitches) % 12
    return len([e for e in pitches if e not in scale_pitches])

def harmonic_closure(melody_path):
    pitches = get_pitches(melody_path)
    key_signature = get_key_signature(melody_path)
    return int(not key_signature.tonic.midi == pitches[-1])

def create_feature_set(melody_path, features, dataset_path):
    # TODO refactor to not recompute this value again
    dataset_mean_pitch = get_mean_pitch_from_dataset(dataset_path)
    dataset_mean_range = get_mean_pitch_range_from_dataset(dataset_path)

    new_pitch_centre = pitch_centre(melody_path, dataset_mean_pitch)
    new_pitch_range = pitch_range(melody_path, dataset_mean_range)
    new_interval_size = interval_size(melody_path)
    new_interval_dissonance = interval_dissonance(melody_path)
    new_chromaticism = chromaticism(melody_path)
    new_harmonic_closure = harmonic_closure(melody_path)

    new_data = [new_pitch_centre, new_pitch_range, new_interval_size,
                new_interval_dissonance, new_chromaticism,
                new_harmonic_closure]

    new_df = pd.DataFrame([new_data])
    new_df.columns = features
    new_df = sm.add_constant(new_df, has_constant='add')
    return new_df

def load_evaluation_model(dataset_path, regression_path=REGRESSION_PATH):
    df = pd.DataFrame(list(RATINGS.items()), columns=['melody', 'rating'])
    dataset_mean_pitch = get_mean_pitch_from_dataset(dataset_path)
    dataset_mean_range = get_mean_pitch_range_from_dataset(dataset_path)

    for filename in glob(regression_path + '/**', recursive=True):
        if filename[filename.rfind("."):] in [".mid", ".midi"]:
            entry = remove_extension(filename)
            pc = pitch_centre(filename, dataset_mean_pitch)
            pr = pitch_range(filename, dataset_mean_range)
            int_size = interval_size(filename)
            int_dissonance = interval_dissonance(filename)
            chro = chromaticism(filename)
            h_closure = harmonic_closure(filename)
            df.loc[df['melody'] == entry, 'pitch_centre'] = pc
            df.loc[df['melody'] == entry, 'pitch_range'] = pr
            df.loc[df['melody'] == entry, 'interval_size'] = int_size
            df.loc[df['melody'] == entry, 'interval_dissonance'] = int_dissonance
            df.loc[df['melody'] == entry, 'chromaticism'] = chro
            df.loc[df['melody'] == entry, 'harmonic_closure'] = h_closure


    X = df[FEATURES]
    X = sm.add_constant(X)
    y = df['rating']
    model = sm.OLS(y, X).fit()
    return model

def evaluate(model, test_melody_path, dataset_path):
    new_df = create_feature_set(test_melody_path,FEATURES, dataset_path)
    predicted_rating = model.predict(new_df).iloc[0]
    return predicted_rating

if __name__ == "__main__":

    dataset_path = sys.argv[1]
    regression_path = sys.argv[2]
    test_melody_path = sys.argv[3]
    model = load_evaluation_model(dataset_path, regression_path)
    print(model.summary().as_latex())
    predicted_rating = evaluate(model, test_melody_path, dataset_path)
    print(f"Predicted rating for the new melody: {predicted_rating}")
