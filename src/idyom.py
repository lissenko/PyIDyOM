import math
import numpy as np
import os
import pickle
import sys
import itertools
import random

from tqdm import tqdm

import matplotlib.pyplot as plt
from itertools import product
from data import CPITCH_VP, score_to_viewpoint

SMALL_VALUE = 1e-15
TMP_BOUND_BEFORE_TREES = 10

def normalize(distribution):
    total = sum(distribution.values())
    for target in distribution:
        distribution[target] /= total
    return distribution

class Idyom:

    def __init__(self, viewpoints=None, quantization=None, use_ppm=None):
        self.viewpoints = viewpoints
        self.quantization = quantization # needed ONLY to take the quantization in generate.py

    def get_information_content(self, probas):
        ics = []
        for p in probas:
            v = p
            if p == 0:
                v += SMALL_VALUE
            ics.append(float(-np.log2(v)))
        return ics

    def get_cross_entropy(self, probas):
        ics = self.get_information_content(probas)
        return sum(ics) / len(ics)

    def q(self, seq, target_elem):
        prob = 1
        for vp_model in self.models.values():
            vp = vp_model.viewpoint
            context = tuple([e[vp] for e in seq])
            elem = target_elem[vp]
            if isinstance(elem, tuple) and None in elem:
                continue
            likelihood = vp_model.getLikelihood(context, elem)
            if likelihood is not None:
                prob *= likelihood
            return prob

    def sample(self, current_score, temperature):

        if len(current_score.pitch) == 0:
            return current_score.key_signature.tonic.midi, 1.0

        sequence = {}
        for vp in self.viewpoints:
            sequence[vp] = tuple(score_to_viewpoint(current_score, vp))
        pitch_sequence = current_score.pitch

        vp_distributions = self.get_distributions(sequence)

        LTM_merged_distribution = self.merge_distributions(vp_distributions)

        ## STM
        STM = Idyom(self.viewpoints, self.quantization)
        STM_representation = {}
        for vp in sequence:
            STM_representation[vp] = [list(sequence[vp])]
        STM.train(STM_representation, [pitch_sequence], is_stm=True)
        STM_vp_distributions = STM.get_distributions(sequence)
        STM_merged_distribution = STM.merge_distributions(STM_vp_distributions)

        LTM_weight = 1.0 / self.get_relative_entropy(LTM_merged_distribution)
        STM_weight = 1.0 / STM.get_relative_entropy(STM_merged_distribution, debug=True)
        weights = [LTM_weight, STM_weight]
        total_weight = sum([w for w in weights])
        final_distribution = {}
        for target in LTM_merged_distribution:
            if target not in STM_merged_distribution:
                final_distribution[target] = LTM_merged_distribution[target]
            else: # Merge
                probas = [LTM_merged_distribution[target], STM_merged_distribution[target]]
                final_distribution[target] = sum([(p*w) for p,w in zip(probas, weights)]) / total_weight

        notes = list(final_distribution.keys())
        final_probabilities = list(final_distribution.values())

        if temperature != 1.0:
            scaled_probabilities = np.log(final_probabilities) / temperature
            scaled_probabilities = np.exp(scaled_probabilities) / np.sum(np.exp(scaled_probabilities))
        else:
            scaled_probabilities = final_probabilities

        sampled_note = random.choices(notes, weights=scaled_probabilities, k=1)[0]

        return sampled_note, final_distribution[sampled_note]

    def get_likelihood_from_sequence(self, sequence, pitch_sequence, use_STM=True):

        size_sequence = len(list(sequence.values())[0])
        predictions = []

        if use_STM:
            STM = Idyom(self.viewpoints, self.quantization)

        for pos in range(size_sequence):
            trimed_sequence = {}
            for vp in sequence:
                trimed_sequence[vp] = tuple(sequence[vp][:pos])

            target = pitch_sequence[pos]

            if target not in self.pitch_alphabet:
                self.pitch_alphabet.append(target)
            for vp in trimed_sequence:
                self.vp_models[vp].update_alphabet(sequence[vp][pos])

            vp_distributions = self.get_distributions(trimed_sequence)
            LTM_merged_distribution = self.merge_distributions(vp_distributions)
            LTM_probability = LTM_merged_distribution[target]
            LTM_weight = 1.0 / self.get_relative_entropy(LTM_merged_distribution)

            if use_STM and pos != 0:
                STM_representation = {}
                for vp in trimed_sequence:
                    STM_representation[vp] = [list(trimed_sequence[vp])]
                # Train the STM
                STM.train(STM_representation, [pitch_sequence[:pos]], is_stm=True)

                # Update alphabet
                if target not in STM.pitch_alphabet:
                    STM.pitch_alphabet.append(target)

                for vp in trimed_sequence:
                    STM.vp_models[vp].update_alphabet(sequence[vp][pos])

                STM_vp_distributions = STM.get_distributions(trimed_sequence)
                STM_merged_distribution = STM.merge_distributions(STM_vp_distributions)

                STM_probability = STM_merged_distribution[target]
                STM_weight = 1.0 / STM.get_relative_entropy(STM_merged_distribution, debug=True)

                weights = [LTM_weight, STM_weight]
                probas = [LTM_probability, STM_probability]
                total_weight = sum([w for w in weights])

                pred = sum([(p*w) for p,w in zip(probas, weights)]) / total_weight

                # Tanguy
                # pred = self.merge_probabilities(probas, weights)

                # simple arithmetic mean
                # pred = np.mean([LTM_probability, STM_probability])

                # Simple geometric mean
                # pred = (LTM_probability * STM_probability)**(1/2)

                predictions.append(pred)

            if use_STM and pos == 0:
                predictions.append(LTM_probability)

            if not use_STM:
                predictions.append(LTM_probability)


        return predictions

    def merge_probabilities(self, probabilities, weights, debug=False):
        """
        Merge probabilities from VP models.
        """
        if len(probabilities) == 1:
            return probabilities[0]

        weight_sum = sum(weights)
        weighted_probas = [p ** w for p, w in zip(probabilities, weights)]
        merged_proba = np.prod(weighted_probas) ** (1 / weight_sum)

        return merged_proba


    def get_distributions(self, different_vps_context):
        distributions = {}
        for vp_model in self.vp_models.values():
            context = different_vps_context[vp_model.viewpoint]
            distrib = vp_model.get_distribution(context, self.pitch_alphabet)
            distributions[vp_model.viewpoint] = distrib
        return distributions

    def plot_distribution(self, distribution, title):
        plt.figure(figsize=(10, 6))
        plt.bar(list(distribution.keys()), list(distribution.values()), color='skyblue')
        plt.title(f'{title}')
        plt.xlabel('Elements')
        plt.ylabel('Probability')
        plt.show()

    def merge_distributions(self, vp_distributions, debug=False):
        distribution = {}
        weights = []
        for vp in vp_distributions: # for each distrib
            weight = 1.0 / self.get_relative_entropy(vp_distributions[vp], debug=debug)

            weights.append(weight)
        total_weight = sum([w for w in weights])

        for target in self.pitch_alphabet:
            probas = []
            for vp in vp_distributions: # for each distrib
                p = vp_distributions[vp][target]
                probas.append(p)

            # Tanguy
            # distribution[target] = self.merge_probabilities(probas, weights)

            distribution[target] = sum([(p*w) for p,w in zip(probas, weights)]) / total_weight

            # Arithmetic mean
            # distribution[target] = np.mean(probas)

        return normalize(distribution)

    def train(self, representation, pitch_representation, is_stm=False):
        # Creer le pitch alphabet ici
        self.pitch_alphabet = list(set(sum(pitch_representation, [])))

        # Find max length to record
        # TODO: Unbounded in practice but maybe set a bound of 30 or so ?
        # max_context_length = max([len(e) for e in pitch_representation])
        max_context_length = TMP_BOUND_BEFORE_TREES

        self.vp_models = {}
        for vp in self.viewpoints:
            self.vp_models[vp] = ViewpointModel(vp, is_stm)
            self.vp_models[vp].train(representation[vp], pitch_representation, max_context_length)

    def save(self, file):

        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = open(file, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, path):

        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          

        self.__dict__.update(tmp_dict) 

    def get_relative_entropy(self, distribution, debug=False):
        max_entropy = self.get_max_entropy(distribution)
        entropy = self.get_entropy(distribution, debug)
        relative_entropy = entropy/max_entropy if max_entropy > 0.0 else 1.0
        return relative_entropy

    def get_max_entropy(self, distribution):
        n = len(distribution)
        max_entropy = np.log2(n)
        return max_entropy

    def get_entropy(self, distribution, debug=False):
        entropy = 0.0
        for p in distribution.values():
            v = p
            if p == 0:
                v += SMALL_VALUE
            entropy -= np.log2(v) * v
        return entropy
            
class ViewpointModel:
    def __init__(self, viewpoint, is_STM):

        self.viewpoint = viewpoint
        self.is_STM = is_STM

    def find_ppm_star_context_length(self, full_context):
        c = []
        for i in range(len(full_context)-1, 0, -1):
            c.insert(0, full_context[i])
            context = tuple(c)
            order = len(context)
            if len(context) <= TMP_BOUND_BEFORE_TREES:
                markov_model = self.markov_models[order]
                if markov_model.t(context) == 1:
                    return order

        # IF no deterministic context -> Find the longest context in db
        for i in range(len(full_context)+1):
            context = tuple(full_context[i:])
            if len(context) <= TMP_BOUND_BEFORE_TREES:
                order = len(context)
                markov_model = self.markov_models[order]
                if context in markov_model.observations:
                    return order

    def get_distribution(self, full_context, basic_domain):
        distribution = {}
        domain = self.get_alphabet()
        
        # Trouver la upper bound
        max_order = self.find_ppm_star_context_length(full_context)
        for target in domain:
            p = self.get_probability(full_context, target, max_order)
            distribution[target] = p

        
        if self.viewpoint != CPITCH_VP:

            # Init
            adapted_distribution = {}
            for e in basic_domain:
                adapted_distribution[e]= 0.0

            context = full_context[-max_order:]

            for domain_elem in domain:
                if domain_elem != None:
                    # phi(c::e)
                    context_target = tuple(list(context) + [domain_elem])
                    proba_to_split = distribution[domain_elem]
                    if context_target in self.mappings:
                        # Tous les c::e
                        every_pitch_context_target = self.mappings[context_target]
                        p = proba_to_split / len(every_pitch_context_target)

                        for seq in every_pitch_context_target:
                            adapted_distribution[seq[-1]] += p

            if not sum(list(adapted_distribution.values())) >= 1.0:

                proba_to_share = 1 - sum(list(adapted_distribution.values()))
                keep = proba_to_share
                right = sum(list(adapted_distribution.values()))
                count_zero = 0
                for p in adapted_distribution:
                    if adapted_distribution[p] == 0.0:
                        count_zero += 1
                if count_zero != 0:
                    proba_to_share /= count_zero
                    for p in adapted_distribution:
                        if adapted_distribution[p] == 0.0:
                            adapted_distribution[p] = proba_to_share

        else:
            adapted_distribution = distribution

        adapted_distribution = normalize(adapted_distribution)
        return normalize(adapted_distribution)

    def get_probability(self, full_context, target, max_order, k=0):
        """
        Interpolated smoothing
        """

        context = full_context[-max_order+k:] if k != max_order else tuple()
        order = len(context)
        markov_model = self.markov_models[order]

        if k == max_order: # end case
            if markov_model.count_context_target(context, target) > 0:
                if self.is_STM:
                    alpha = markov_model.alpha_x(context, target)
                else:
                    # C
                    alpha = markov_model.alpha_c(context, target)
                v = alpha
            elif len(self.get_alphabet()) > 0:
                v = 1 / len(self.get_alphabet())
            return v

        if self.is_STM:
            alpha = markov_model.alpha_x(context, target)
            gamma = markov_model.gamma_x(context)
        else:
            # C
            alpha = markov_model.alpha_c(context, target)
            gamma = markov_model.gamma_c(context)
        return alpha + gamma * self.get_probability(full_context, target, max_order, k+1)

    def get_alphabet(self):
        return self.markov_models[0].alphabet

    def update_alphabet(self, e):
        if e not in self.markov_models[0].alphabet:
            self.markov_models[0].alphabet.append(e)

    def train(self, viewpoint_sequences, pitch_sequences, max_context_length):
        self.markov_models = {}
        self.mappings = {}
        for order in range(max_context_length, -1, -1):
            mapping = True if self.viewpoint != CPITCH_VP else False
            self.markov_models[order] = MarkovModel(order)
            mappings = self.markov_models[order].train(viewpoint_sequences, pitch_sequences, mapping)
            for context_target in mappings:
                self.mappings[context_target] = mappings[context_target]


class MarkovModel:
    def __init__(self, order):
        self.order = order

        self.observations = {}
        self.alphabet = []

    def t(self, context):
        """
        The total number of symbol types that have occured with non-zero frequency in context
        """
        count = 0
        if context in self.observations:
            count = len(self.observations[context])
        return count

    def t_k(self, k, context):
        """
        The total number of symbol types that have occurred exactly k
        times in the context
        """
        count = 0
        if context in self.observations:
            for symbol in self.observations[context]:
                if self.observations[context][symbol] == k:
                    count += 1
        return count

    def count_context(self, context):
        """
        sum for all target after the context
        """
        count = 0
        if context in self.observations:
            count = sum(self.observations[context].values())
        return count

    def count_context_target(self, context, target):
        count = 0
        if context in self.observations and target in self.observations[context]:
            count = self.observations[context][target]
        return count

    def gamma_x(self, context):
        return (self.t_k(1, context) + 1) / (self.count_context(context) + self.t_k(1, context) + 1)

    def alpha_x(self, context, target):
        return (self.count_context_target(context, target)) / (self.count_context(context) + self.t_k(1, context) + 1)

    def gamma_c(self, context):
        return (self.t(context)) / (self.count_context(context) + self.t(context))

    def alpha_c(self, context, target):
        # POURQUOI CA ME DIT DIVISION PAR 0 -> J'ai forcement une valeur pour le contexte non ?
        return (self.count_context_target(context, target)) / (self.count_context(context) + self.t(context))

    def train(self, viewpoint_sequences, pitch_sequences, mapping):
        # Reset the observations
        self.observations = {}

        order = self.order

        mappings = {}
        for i in range(len(viewpoint_sequences)):
            sequence = viewpoint_sequences[i]
            size_seq = len(sequence)
            for j in range(order, size_seq):
                target = sequence[j]
                if target not in self.alphabet:
                    self.alphabet.append(target)
                context = tuple([sequence[j-k] for k in range(order, 0, -1)])
                if context not in self.observations:
                    self.observations[context] = {target: 1}
                else:
                    if target not in self.observations[context]:
                        self.observations[context][target] = 1
                    else:
                        self.observations[context][target] += 1


                if mapping:

                    context_target = tuple(list(context) + [target])

                    pitch_sequence = pitch_sequences[i]
                    pitch_context = tuple([pitch_sequence[j-k] for k in range(self.order, 0, -1)])
                    pitch_target = pitch_sequence[j]
                    pitch_context_target = tuple(list(pitch_context) + [pitch_target])

                    if context_target not in mappings:
                        mappings[context_target] = [pitch_context_target]
                    else:
                        if pitch_context_target not in mappings[context_target]:
                            mappings[context_target].append(pitch_context_target)
        return mappings 

