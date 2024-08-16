import numpy as np
from music21 import converter, note, scale, pitch

try:
    import matplotlib.pyplot as plt
    plot = True
except ImportError:
    plot = False


def getNotesFromMidi(midiFile):
    """
    Compute a list of Note objects from a midi file.
    ----------
    midiFile : str
        Filepath of the midi file.

    Returns
    -------
    notes : list of Note.
        List of Note objects.
    """

    notes = []
    rests = []
    midi_stream = converter.parse(midiFile)
    score = midi_stream.flatten()
    resolution = midi_stream.highestTime

    time_signature = score.getElementsByClass('TimeSignature')[0]
    if not time_signature:
        time_signature = stream.TimeSignature('4/4')

    key_signature = score.getElementsByClass('KeySignature')
    if not key_signature:
        # key_signature = score.analyze('key')
        key_signature = score.analyze('key.krumhanslschmuckler')
    else:
        key_signature = key_signature[0]

    for element in score.notesAndRests:
        if isinstance(element, note.Note):
            # If it's a note
            tick_duration = element.duration.quarterLength * resolution
            note_obj = Note(element.volume.velocity,
                            element.pitch.midi,
                            element.offset * resolution,
                            (element.offset + element.duration.quarterLength) * resolution)
            notes.append(note_obj)

        elif isinstance(element, note.Rest):
            tick_duration = element.duration.quarterLength * resolution
            start = element.offset * resolution
            end = (element.offset + element.duration.quarterLength) * resolution
            rest = (start, end)
            rests.append(rest)

    return notes, rests, resolution, time_signature, key_signature


def getRepresentationFromNotes(filename, notes, quantization):
    """
    Get a list of notes and return the viewpoint (pitch, durations) representation
    ----------
    notes : list of Notes
        List of Note objects.
    quantization : int (optional)
        Quantization factor, it is the number of ticks in 1 beat. Default is 96

    Returns
    -------
    pitch : list of int
        List of the midi pitch cotained in the file.
    duration : list of int
        List of the quantized (in ticks not time) durations of the notes
    onset : list of int
        List of the quantized (in ticks not time) onsets of the notes
    velocity : list of int
        List of velocities (intensity).
    silenceBegining : int
        The number of ticks of silence at the beging of the file (if anacrouse).
    """
    notes, rests, ticks_per_beat, time_signature, key_signature = notes

    pulses = time_signature.numerator
    barlength = ticks_per_beat * pulses
    barlength = round((quantization * barlength) / ticks_per_beat)

    pitch = []
    duration = []
    onset = []
    velocity = []
    deltast = []

    for i in range(len(notes)-1):
        if notes[i+1].start < notes[i].end:
            print("Erreur avec: ", filename)
            raise RuntimeError("This Midi is polyphonic, I cannot handle this.")

        # If there is a rest before the next note
        isRest = False
        durationUntilNextNote = 0
        for rest_start, rest_end in rests:
            if rest_start > notes[i].start and rest_start < notes[i+1].start:
                durationUntilNextNote = rest_start - notes[i].start
                isRest = True
                break


        if not isRest:
            durationUntilNextNote = notes[i+1].start - notes[i].start
        pitch.append(notes[i].pitch)
        dur = round(quantization*durationUntilNextNote/ticks_per_beat)
        duration.append(dur)

        onset.append(round((quantization * notes[i].start)/ticks_per_beat))
        velocity.append(notes[i].velocity)

    pitch.append(notes[-1].pitch)
    velocity.append(notes[-1].velocity)

    # Find the last duration
    candidate_durations = [quantization * m for m in [1/16, 1/8, 1/4, 1/2, 1, 2, 4]]
    last_duration = round(quantization * (notes[-1].end - notes[-1].start) / ticks_per_beat)
    # Find the closest duration in candidate_durations
    closest_duration_index = min(range(len(candidate_durations)), key=lambda i: abs(candidate_durations[i] - last_duration))
    closest_duration = candidate_durations[closest_duration_index]
    duration.append(closest_duration)
    onset.append(round((quantization * notes[-1].start)/ticks_per_beat))
    silenceBegining = round(quantization*notes[0].start/ticks_per_beat)

    # The first note has no deltast
    deltast.append(0)
    for i in range(1, len(notes)):
        # intervalle de temps entre la fin d'un event et l'onset du prochain
        deltast.append(round(-(quantization * (notes[i-1].end - notes[i].start)) / ticks_per_beat))

    return pitch, duration, onset, velocity, silenceBegining, deltast, barlength, pulses, key_signature

def readMidi(file, quantization):
    """
    Open the passed midi file and returns the viewpoint (pitch, duration) representation.
    Parameters
    ----------
    file : str
        Path to the midi file. It needs to be monophonic.
    quantization : int (optional)
        Quantization factor, it is the number of ticks in 1 beat. Default is 96

    Returns
    -------
    pitch : list of int
        List of the midi pitch cotained in the file.
    duration : list of int
        List of the quantized (in ticks not time) durations of the notes.

    velocity : list of int
        List of velocities (intensity).
    silenceBegining : int
        The number of ticks of silence at the beging of the file (if anacrouse).
    """
    return getRepresentationFromNotes(file, getNotesFromMidi(file), quantization)


class Note(object):
    """A simple Note object.
    Parameters
    ----------
    velocity : int
        Note velocity.
    pitch : int
        Note pitch, as a MIDI note number.
    start : float
        Note on time, absolute, in ticks.
    end : float
        Note off time, absolute, in ticks.
    """

    def __init__(self, velocity, pitch, start, end):
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def get_duration(self):
        """Get the duration of the note in ticks."""
        return self.end - self.start

    def __repr__(self):
        return 'Note(start={}, end={}, pitch={}, velocity={})'.format(
                self.start, self.end, self.pitch, self.velocity)

class Score():
    """This class is used to manage midi data from the dataset.
    It uses the module mido to transform midi files in numpy arrays. 
    Thus, a score object can be created from a midi file.
    Attributes
    ----------
    name : str
        Name of the original midi file.
    pitch : list of int
        List of the midi pitches.
    duration : list of int
        List of the durations of the notes.
    velocity: List of int
    List of the velocities of the notes (intensity).
    silenceBegining : int
        Number of ticks of silence at the bigining of the file.
    quantization : int
        Ticks quantization per beat. 96 by default.
    """

    def __init__(self, fileName, quantization):
        self.name = fileName
        self.quantization = quantization
        self.pitch, self.duration, self.onset, self.velocity, self.silenceBegining, self.deltast, self.barlength, self.pulses, self.key_signature = readMidi(fileName, quantization)

    def new_score_from_pitches(self, new_pitches):

        new_score = Score(self.name, self.quantization)
        new_score.pitch = new_pitches

        # Iterate over all other attributes and copy them from the original score
        for attr in vars(self):
            if attr != 'pitch':
                setattr(new_score, attr, getattr(self, attr))

        return new_score

    def size(self):
        return len(self.pitch)

    def keysig(self):
        key_numeric = self.key_signature.tonic.midi % 12 - 6 if self.key_signature.mode == 'major' else (self.key_signature.tonic.midi + 3) % 12 - 6
        return key_numeric

    def mode(self):
        return 0 if self.key_signature.mode == 'major' else 9

    def cpint(self):
        return [None] + list(np.diff(self.pitch))

    def pitchClass(self):
        return np.array(self.pitch) % 12

    def contour(self):
        result = [0 for _ in range(self.size()-1)]
        for i in range(1, self.size()):
            if self.pitch[i-1] > self.pitch[i]:
                result[i-1] = -1
            elif self.pitch[i-1] < self.pitch[i]:
                result[i-1] = 1
        return [None] + result

    def referent(self):
        return list(np.array([self.key_signature.tonic.midi for _ in range(self.size())]) % 12)

    def inscale(self):
        scale_name = self.key_signature.getScale()
        scale_pitches = np.array([e.midi for e in scale_name.getPitches()]) % 12
        pitches = np.array(self.pitch) % 12
        return [True if p in scale_pitches else False  for p in pitches]

    def intFromTonic(self):
        tonic = self.key_signature.tonic.midi % 12
        base = np.array(self.pitch) % 12
        intervals = []
        for n in base:
            interval = 0
            while n != tonic:
                n += 1
                n %= 12
                interval += 1
            intervals.append(interval)
        return intervals

    def intFromFip(self):
        return list(np.array(self.pitch) - self.pitch[0])

    def intFromFib(self):
        bar_duration = self.quantization * self.pulses
        fib_notes = []
        fib_note = 0
        for pitch, onset, deltast in zip(self.pitch, self.onset, self.deltast):
            isFib = ((onset) % bar_duration) == 0
            isFibWithRest = ((onset - deltast) % bar_duration) == 0
            isFibWithAnacrouse = (onset == self.silenceBegining) and ((onset - self.silenceBegining) % bar_duration) == 0
            if isFib or isFibWithRest or isFibWithAnacrouse:
                fib_note = pitch
            fib_notes.append(fib_note)
        return list(np.array(self.pitch) - np.array(fib_notes))

    def posinbar(self):
        bar_duration = self.quantization * self.pulses
        bars_count = []
        bar_idx = 0
        for pitch, onset, deltast in zip(self.pitch, self.onset, self.deltast):
            isFib = ((onset) % bar_duration) == 0
            isFibWithRest = ((onset - deltast) % bar_duration) == 0
            isFibWithAnacrouse = (onset == self.silenceBegining) and ((onset - self.silenceBegining) % bar_duration) == 0
            if isFib or isFibWithRest or isFibWithAnacrouse:
                bar_idx += 1
                bars_count.append(0)
            bars_count[bar_idx-1] += 1
        posinbar = []
        for count in bars_count:
            for i in range(count):
                posinbar.append(i)
        return posinbar

    def fib(self):
        bar_duration = self.quantization * self.pulses
        fib = []
        for pitch, onset, deltast in zip(self.pitch, self.onset, self.deltast):
            isFib = ((onset) % bar_duration) == 0
            isFibWithRest = ((onset - deltast) % bar_duration) == 0
            isFibWithAnacrouse = (onset == self.silenceBegining) and ((onset - self.silenceBegining) % bar_duration) == 0
            fib.append(isFib or isFibWithRest or isFibWithAnacrouse)
        return fib

    def ioi(self):
        return [None] + list(np.diff(self.onset))

    def durRatio(self):
        ratios = []
        for i in range(1, self.size()):
            ratios.append(self.duration[i]/self.duration[i-1])
        return [None] + ratios

    def tactus(self):
        return [True if (self.onset[i] % (self.barlength/self.pulses)) == 0 else False for i in range(self.size())]

    def threadedTactus(self):
        result = []
        for t, i in zip(self.tactus(), self.cpint()):
            res = i if t else None
            result.append(res)
        return result

    def threadedBar(self):
        result = []
        for fib, i in zip(self.fib(), self.cpint()):
            res = i if fib else None
            result.append(res)
        return result

    def getLength(self):
        """Returns the length in time beats."""

        return (np.sum(self.duration)+self.silenceBegining)

    def getData(self):
        """
        Returns the data as a list of pitch and a list of durations
        """
        return self.pitch, self.duration

    def getOnsets(self, sr=64, tempo=120):
        """
        Returns the onsets representation, useful for EEG analysis for instance.
        ----------
        sr : int
            Sampling rate for the returned signal (64 by default).
        tempo : int
            Tempo you want to render the signal at (default 120.

        Returns
        -------
        onsets : list of int
            Signal sampled at sr Hz containing 0s and 1s. Ones means that there is a note.
        """
        onsets = []
        onsets.extend([0]*round(self.silenceBegining*sr*60/tempo))
        for d in self.duration:
            onsets.append(1)
            onsets.extend([0]*(round(d*sr*60/tempo)-1))

        return onsets

    def getPianoRoll(self):
        """
        Returns the numpy array containing the pianoRoll. The pitch is render in a 128 dimensions axis.

        Returns
        -------
        pianoRoll : numpy array
            Numpy array containing the pianoroll the pitch dimension shape is always 128.
        """

        mat = np.zeros((128, int(self.getLength())))

        tick = self.silenceBegining
        for i in range(len(self.pitch)):
            mat[self.pitch[i], tick:tick+self.duration[i]] = self.velocity[i]
            tick += self.duration[i]

        return mat


    def plot(self):
        """Plots the pianoRoll representation."""

        if plot == False:
            print("you cannot plot anything as matplotlib is not available")
            return
        minPitch = min(self.pitch)
        maxPitch = max(self.pitch)
        pitchRange = maxPitch - minPitch

        mat = np.zeros((int(pitchRange)+1, int(self.getLength())))

        tick = self.silenceBegining
        for i in range(len(self.pitch)):
            mat[self.pitch[i]-minPitch, tick:tick+self.duration[i]] = self.velocity[i]
            tick += self.duration[i]

        plt.imshow(mat, aspect='auto', origin='lower')
        plt.xlabel('time (beat)')
        plt.ylabel('midi note')
        plt.grid(visible=True, axis='y')
        plt.yticks(list(range(maxPitch-minPitch)), list(range(minPitch, maxPitch)))
        color = plt.colorbar()
        color.set_label('velocity', rotation=270)
        plt.show()
