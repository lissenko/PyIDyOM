import numpy as np
import math
import logging
from data import get_representations
from idyom import Idyom
from sklearn.metrics import mean_squared_error

from data import Data, AVAILABLE_VIEWPOINTS, CPITCH_VIEWPOINTS, DELIMITER, CPITCH_VP, get_representations
from variance_comparison import model_comparison

# Set up logging configuration
logging.basicConfig(
    filename="feature_selection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def is_improvement(candidate_fit, best_fit):
    candidate_fit = tuple(candidate_fit.values())
    best_fit = tuple(best_fit.values())
    improvement = np.mean(candidate_fit) < np.mean(best_fit)
    if improvement:
        print("OUI, avec variance: ", 1 - np.mean(candidate_fit))
    return improvement

def test_candidate_set(idyom, test_data, test_files, candidate_set, reference_data, use_stm):
    candidate_fit = {}
    test_representation, pitch_test_representation = get_representations(test_data, test_files, candidate_set)
    for num_score, name in zip(range(len(test_files)), test_data.scores):
        pitch_sequence = pitch_test_representation[num_score]
        sequence = {}
        for vp in idyom.viewpoints:
            sequence[vp] = test_representation[vp][num_score]
        probas = idyom.get_likelihood_from_sequence(sequence, pitch_sequence, use_stm)
        ics = idyom.get_information_content(probas)

        R, R_squared, metric, F_statistic, p_value = model_comparison(reference_data[name], ics)
        metric = 1 - metric

        candidate_fit[name] = metric
    return candidate_fit

"""
Feature selection to fit the behavioral data from Manzara
"""
def manzara_feature_selection(all_data, test_data, files, test_files, standard_features, use_stm, reference_data):
    logger = logging.getLogger()

    # Use every combination possible
    features = []
    for feature in standard_features:
        if feature in CPITCH_VIEWPOINTS:
            features.append(feature)

    for f1 in standard_features:
        for f2 in standard_features:
            if f1 != f2 and f"{f1}&{f2}" not in features and f"{f2}&{f1}" not in features:
                if f1 in CPITCH_VIEWPOINTS or f2 in CPITCH_VIEWPOINTS:
                    features.append(f"{f1}&{f2}")
    print("Every features: ", features)

    best_set = []
    best_fit = {}
    for score_name in reference_data:
        best_fit[score_name] = float('inf')
    improvement = True
    while improvement:
        improvement = False
        # consider every addition
        for candidate_feature in features:
            if candidate_feature not in best_set:
                candidate_set = best_set.copy()
                candidate_set.append(candidate_feature)
                print("Addition: ", candidate_set)
                # TRAIN
                representation, pitch_representation = get_representations(all_data, files, candidate_set)
                idyom = Idyom(candidate_set)
                idyom.train(representation, pitch_representation)
                candidate_fit = test_candidate_set(idyom, test_data, test_files, candidate_set, reference_data, use_stm)
                if is_improvement(candidate_fit, best_fit):
                    best_set = candidate_set
                    best_fit = candidate_fit
                    improvement = True
                    logger.info(f"{best_fit} for features {best_set}")

        # consider every deletion
        for candidate_feature in best_set:
            candidate_set = best_set.copy()
            candidate_set.remove(candidate_feature)
            if len(candidate_set) > 0:
                # TRAIN
                print("Deletion: ", candidate_set)
                representation, pitch_representation = get_representations(all_data, files, candidate_set)
                idyom = Idyom(candidate_set)
                idyom.train(representation, pitch_representation)
                candidate_fit = test_candidate_set(idyom, test_data, test_files, candidate_set, reference_data, use_stm)
                if is_improvement(candidate_fit, best_fit):
                    best_set = candidate_set
                    best_fit = candidate_fit
                    improvement = True
                    logger.info(f"{best_fit} for features {best_set}")

    print("The best set is: ", best_set)
    print("With fit: ", best_fit)
    print("avec variance: ", 1 - np.mean(best_fit))

def cross_validation(all_data, viewpoints, folds, files, use_stm):

    if folds > len(files):
        raise ValueError("Cannot process with k_fold greater than number of files. Please use -k options to specify a smaller k for cross validation.")
    k_fold = len(files) // (folds-1) # number of files to evaluate

    cross_entropies = []

    for i in range(math.ceil(len(files)/k_fold)):
        train_files = files[:i*k_fold] + files[(i+1)*k_fold:]
        eval_files = files[i*k_fold:(i+1)*k_fold]

        print(f"==========\n  FOLD {i}  \n==========")

        idyom = Idyom(viewpoints)
        print('____ Training ____')
        representation, pitch_representation = get_representations(all_data, train_files, idyom.viewpoints)
        idyom.train(representation, pitch_representation)

        eval_data = Data(eval_files, all_data.quantization)
        eval_data.parse(augment=False) # Do not augment the evaluation data

        representation, pitch_representation = get_representations(eval_data, eval_files, idyom.viewpoints)

        count = len(eval_data.scores)
        for num_score, name in zip(range(count), eval_data.scores):
            sequence = {}
            for vp in idyom.viewpoints:
                sequence[vp] = representation[vp][num_score]
            pitch_sequence = pitch_representation[num_score]
            probas = idyom.get_likelihood_from_sequence(sequence, pitch_sequence, use_stm)
            cross_entropy = idyom.get_cross_entropy(probas)
            cross_entropies.append(cross_entropy)

    mean_cross_entropy = np.mean(cross_entropies)
    return mean_cross_entropy

def uncertainty_feature_selection(all_data, files, standard_features, use_stm, folds):
    logger = logging.getLogger()


    # Use every combination possible
    features = []
    for feature in standard_features:
        if feature in CPITCH_VIEWPOINTS:
            features.append(feature)

    for f1 in standard_features:
        for f2 in standard_features:
            if f1 != f2 and f"{f1}&{f2}" not in features and f"{f2}&{f1}" not in features:
                if f1 in CPITCH_VIEWPOINTS or f2 in CPITCH_VIEWPOINTS:
                    features.append(f"{f1}&{f2}")
    print("Every features: ", features)

    best_set = []
    best_fit = float('inf')

    improvement = True
    while improvement:
        improvement = False
        # consider every addition
        for candidate_feature in features:
            if candidate_feature not in best_set:
                candidate_set = best_set.copy()
                candidate_set.append(candidate_feature)
                print("Addition: ", candidate_set)

                candidate_fit = cross_validation(all_data, candidate_set, folds, files, use_stm)

                if candidate_fit < best_fit:
                    best_set = candidate_set
                    best_fit = candidate_fit
                    improvement = True
                    logger.info(f"Cross_entropy={best_fit} for features {best_set}")

        # consider every deletion
        for candidate_feature in best_set:
            candidate_set = best_set.copy()
            candidate_set.remove(candidate_feature)
            if len(candidate_set) > 0:
                print("Delete: ", candidate_set)

                candidate_fit = cross_validation(all_data, candidate_set, folds, files, use_stm)

                if candidate_fit < best_fit:
                    best_set = candidate_set
                    best_fit = candidate_fit
                    improvement = True
                    logger.info(f"Cross_entropy={best_fit} for features {best_set}")

    print("The best set is: ", best_set)
    print("With cross_entropy: ", best_fit)
