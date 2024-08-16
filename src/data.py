import numpy as np
import os

from midi import Score
from music21 import interval
from tqdm import tqdm

# Basic 
CPITCH_VP = "cpitch"
DUR_VP = "dur"
VEL_VP = "vel"
ONSET_VP = "onset"
DELTAST_VP = "deltast"
BARLENGTH_VP = "barlength"
PULSES_VP = "pulses"
KEYSIG_VP = "keysig"
MODE_VP = "mode"

# Derived
CPINT_VP = "cpint"
CPITCH_CLASS_VP = "cpitch-class"
CPCINT_VP = "cpcint"
CONTOUR_VP = "contour"
REFERENT_VP = "referent"
INSCALE_VP = "inscale"
CPINTFREF_VP = "cpintfref"
CPINTFIP_VP = "cpintfip"
CPINTFIB_VP = "cpintfib"
POSINBAR_VP = "posinbar"
IOI_VP = "ioi"
DUR_RATIO_VP = "dur-ratio"
TESSITURA_VP = "tessitura"

# Test
TACTUS_VP = "tactus"
FIB_VP = "fib"

# Threaded
THRTACTUS_VP = "thrtactus"
THRBAR_VP = "thrbar"

NULL_SYMBOL = None

DELIMITER = '&'

NOT_USELESS_VIEWPOINTS = [
        CPITCH_VP,
        DUR_VP,
        ONSET_VP,
        KEYSIG_VP,
        MODE_VP,
        CPITCH_CLASS_VP,
        CPINT_VP,
        CPCINT_VP,
        CONTOUR_VP,
        REFERENT_VP,
        INSCALE_VP,
        CPINTFREF_VP,
        CPINTFIP_VP,
        CPINTFIB_VP,
        POSINBAR_VP,
        IOI_VP,
        DUR_RATIO_VP,
        TESSITURA_VP,
        TACTUS_VP,
        FIB_VP,
        THRTACTUS_VP,
        THRBAR_VP
        ]

AVAILABLE_VIEWPOINTS = [
        CPITCH_VP,
        DUR_VP,
        ONSET_VP,
        VEL_VP,
        DELTAST_VP,
        BARLENGTH_VP,
        PULSES_VP,
        KEYSIG_VP,
        MODE_VP,
        CPITCH_CLASS_VP,
        CPINT_VP,
        CPCINT_VP,
        CONTOUR_VP,
        REFERENT_VP,
        INSCALE_VP,
        CPINTFREF_VP,
        CPINTFIP_VP,
        CPINTFIB_VP,
        POSINBAR_VP,
        IOI_VP,
        DUR_RATIO_VP,
        TESSITURA_VP,
        TACTUS_VP,
        FIB_VP,
        THRTACTUS_VP,
        THRBAR_VP
        ]

CPITCH_VIEWPOINTS = [CPITCH_VP, CPINT_VP, CPINTFREF_VP, CPITCH_CLASS_VP,
                     CPCINT_VP, CONTOUR_VP, INSCALE_VP, CPINTFIP_VP,
                     CPINTFIB_VP, TESSITURA_VP, THRTACTUS_VP, THRBAR_VP]

def score_to_viewpoint(score, viewpoint):

    # Basic
    if DUR_VP == viewpoint:
        return score.duration

    if CPITCH_VP == viewpoint:
        return score.pitch

    if VEL_VP == viewpoint:
        return score.velocity

    if ONSET_VP == viewpoint:
        return score.onset

    if DELTAST_VP == viewpoint:
        return score.deltast

    if BARLENGTH_VP == viewpoint:
        return [score.barlength for _ in range(score.size())]

    if PULSES_VP == viewpoint:
        return [score.pulses for _ in range(score.size())]

    if KEYSIG_VP == viewpoint:
        return [score.keysig() for _ in range(score.size())]

    if MODE_VP == viewpoint:
        return [score.mode() for _ in range(score.size())]

    # Derived 
    if CPINT_VP == viewpoint:
        return score.cpint()

    if CPITCH_CLASS_VP == viewpoint:
        return score.pitchClass()

    if CPCINT_VP == viewpoint:
        return [NULL_SYMBOL] + list(np.diff(score.pitchClass()))

    if CONTOUR_VP == viewpoint:
        return score.contour()

    if REFERENT_VP == viewpoint:
        return score.referent()

    if INSCALE_VP == viewpoint:
        return score.inscale()

    if CPINTFREF_VP == viewpoint:
        return score.intFromTonic()

    if CPINTFIP_VP == viewpoint:
        return score.intFromFip()

    if CPINTFIB_VP == viewpoint:
        return score.intFromFib()

    if POSINBAR_VP == viewpoint:
        return score.posinbar()

    if IOI_VP == viewpoint:
        return score.ioi()

    if DUR_RATIO_VP == viewpoint:
        return score.durRatio()

    if TESSITURA_VP == viewpoint:
        pitches = []
        mean_pitch = 70
        std_dev = 3.5945754733207314 
        tessitura = []
        for pitch in score.pitch:
            if pitch >= mean_pitch - std_dev and pitch <= mean_pitch + std_dev:
                tessitura.append(0)
            elif pitch < mean_pitch - std_dev:
                tessitura.append(-1)
            elif pitch > mean_pitch + std_dev:
                tessitura.append(1)
        return tessitura

    if TACTUS_VP == viewpoint:
        return score.tactus()

    if FIB_VP == viewpoint:
        return score.fib()

    # Threaded

    if THRTACTUS_VP == viewpoint:
        return score.threadedTactus()

    if THRBAR_VP == viewpoint:
        return score.threadedBar()

    # Linked

    if DELIMITER in viewpoint:
        vps = [vp for vp in viewpoint.split(DELIMITER)]
        representations = [score_to_viewpoint(score, vp) for vp in vps]
        representation = []
        for i in range(score.size()):
            representation.append(tuple([rep[i] for rep in representations]))
        return representation

def get_representations(data, files, vps):
    representation = data.get_viewpoint_representation(files, vps)
    if CPITCH_VP in vps:
        pitch_representation = representation[CPITCH_VP]
    else:
        pitch_representation = data.get_viewpoint_representation(files, [CPITCH_VP])[CPITCH_VP]
    return representation, pitch_representation


class Data:
    def __init__(self, files, quantization):
        self.files = files
        self.quantization = quantization
        self.scores = {}


    def parse(self, augment=False):
        print(f"____ Get score representation ____")
        for file_path in tqdm(self.files, total=len(self.files)):
            score = Score(file_path, self.quantization)
            self.scores[file_path] = [score]
        if augment:
            print(f"\n____ Data Augmentation ____")
            count = len(self.scores)
            for score_name in tqdm(self.scores, total=count):
                score = self.scores[score_name][0]
                for t in range(-5, 7):
                    if t != 0:
                        new_score = score.new_score_from_pitches(list(np.array(score.pitch) + t))
                        transposed_key = new_score.key_signature.transpose(t)
                        new_score.key_signature = transposed_key
                        self.scores[score_name].append(new_score)

    def get_viewpoint_representation(self, score_names, viewpoints):
        representation = {}

        for vp in viewpoints:
            representation[vp] = []
            for score_name in score_names:
                scores = self.scores[score_name]
                for score in scores:
                    representation[vp].append(score_to_viewpoint(score, vp))
                    # Append fixed pitch reprensentation

        # for k, v in zip(representation.keys(), representation.values()):
        #     print(f'{k}: {v}')

        return representation


