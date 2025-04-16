import random
import json

import pandas as pd
import numpy as np

# circuits are pasted in the main.py file
from circuit_maker import Circuit, CircuitFactory


def simple_test():
    # Circuit Factory Example 
    print("Circuit Factory Example") 

    # Parameters for the circuits
    num_inputs = 4
    num_target_classes = 3 # e.g., 3 classes for a simple problem
    num_gates = 5
    num_circuits = 5 # Keep small for readability

    # Create a CircuitFactory
    circuit_factory = CircuitFactory(num_inputs, num_target_classes, num_gates)

    # Build circuits
    circuits = circuit_factory.build_circuits(num_circuits)
    print(f"Built {num_circuits} random circuits.")
    print(f"Each circuit has {circuits[0].total_nodes} total nodes ({num_inputs} inputs + {num_gates} gates).")
    print(f"Targets have {num_target_classes} classes.")

    # Sample Dataset (One-hot encoded)
    data = [[random.choice([0, 1]) for _ in range(num_inputs)] for _ in range(50)]
    targets = []
    for _ in range(50):
        target_idx = random.randrange(num_target_classes)
        one_hot = [0] * num_target_classes
        one_hot[target_idx] = 1
        targets.append(one_hot)

    print(f"Generated sample dataset with {len(data)} samples.")


    json_file = "simple_circuits_scores.json"
    circuit_factory.write_report(json_file, circuits, data, targets)
    print(f"\nCircuit data written to {json_file}")


def digits_test():
    """
    Test the Circuit class with the digits dataset and the new scoring.

    at random chance the accuracy is 0.1
    always guess 0 gives an accuracy of 0.9

    so we need a better dataset to test the circuits.
    """
    print("--- Digits Test ---")
    # Parameters for the circuits
    num_inputs = 64
    num_target_classes = 10 # MNIST digits 0-9
    num_gates = 150 # Keep small initially for testing
    num_circuits = 100 # Keep small for testing

    # Create a CircuitFactory
    circuit_factory = CircuitFactory(num_inputs, num_target_classes, num_gates)

    # Build circuits
    circuits = circuit_factory.build_circuits(num_circuits)
    print(f"Built {num_circuits} random circuits.")
    print(f"Each circuit has {circuits[0].total_nodes} total nodes ({num_inputs} inputs + {num_gates} gates).")

    

    # Load the dataset
    try:
        df = pd.read_csv('datasets/digits.csv')
        # Ensure data is binary (0 or 1) if it isn't already
        # For digits dataset, it's often grayscale 0-16. Binarize it.
        data = (df.iloc[:, :64].values > 8).astype(int) # Binarize threshold at 8 (midpoint)
        targets = df.iloc[:, 64:].values # Assume targets are already one-hot encoded
        print(f"Loaded and binarized digits dataset. Shape: data={data.shape}, targets={targets.shape}")
        if targets.shape[1] != num_target_classes:
             print(f"Warning: Dataset target columns ({targets.shape[1]}) != num_target_classes ({num_target_classes}). Check dataset format.")
             # Adjust num_target_classes if necessary, or raise error
             # num_target_classes = targets.shape[1]
             # print(f"Adjusted num_target_classes to {num_target_classes}")
    except FileNotFoundError:
        print("Error: `datasets/digits.csv` not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return
    
    # update the circuits with their scores
    circuit_factory.update(circuits, data, targets)

    json_file = "digits_circuits_detailed_scores.json"
    circuit_factory.write_report(json_file, circuits, data, targets)
    print(f"\nCircuit data written to {json_file}")



def iris_binary_test():
    """
    Test the Circuit class with the iris binary dataset.

    featues: 4 x 4 binary features
    target: 3 x 1 hot encoded target

    SepLen_1,SepLen_2,SepLen_3,SepLen_4,
    SepWid_1,SepWid_2,SepWid_3,SepWid_4,
    PetLen_1,PetLen_2,PetLen_3,PetLen_4,
    PetWid_1,PetWid_2,PetWid_3,PetWid_4,
    
    target_0,target_1,target_2
    
    0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,1,0,0
    1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0

    ## How well does a decision tree classifier do on this dataset?

    Accuracy: 0.93
    Classification Report:
        precision    recall  f1-score   support

    0       0.83      1.00      0.91        10
    1       1.00      0.89      0.94         9
    2       1.00      0.91      0.95        11

    ## How well circuits do?

    Scoring Circuit 98...
        Circuit 98: Score Matrix Shape = (26, 3)
        Max accuracy per class: [0.867 0.86  0.96 ]
        Best node index per class: [13 14 15]
    Scoring Circuit 99...
        Circuit 99: Score Matrix Shape = (26, 3)
        Max accuracy per class: [0.867 0.86  0.96 ]
        Best node index per class: [13 14 15]


    """
    print("--- Iris Binary Test ---")
    random.seed(42)  # For reproducibility
    np.random.seed(42)

    # Parameters for the circuits
    num_inputs = 16  # 4 binary features (4x4) = 16 inputs
    num_target_classes = 3  # Iris has 3 classes
    num_gates = 50  # Adjust as needed
    num_circuits = 500  # Number of circuits to build

    # Create a CircuitFactory
    circuit_factory = CircuitFactory(num_inputs, num_target_classes, num_gates)

    # Build circuits
    circuits = circuit_factory.build_circuits(num_circuits)
    print(f"Built {num_circuits} random circuits.")
    print(f"Each circuit has {circuits[0].total_nodes} total nodes ({num_inputs} inputs + {num_gates} gates).")

    # Load the dataset
    try:
        df = pd.read_csv('datasets/iris_binary.csv')
        data = df.iloc[:, :-num_target_classes].values  # Features
        targets = df.iloc[:, -num_target_classes:].values  # One-hot encoded target
        print(f"Loaded iris binary dataset. Shape: data={data.shape}, targets={targets.shape}")
        if targets.shape[1] != num_target_classes:
            print(f"Warning: Dataset target columns ({targets.shape[1]}) != num_target_classes ({num_target_classes}). Check dataset format.")
    except FileNotFoundError:
        print("Error: `datasets/iris_binary.csv` not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return

    # update the circuits with their scores
    circuit_factory.update(circuits, data, targets)

    # write report before cleaning up the circuits
    json_file = "iris_binary_circuits_detailed_scores_before_clean_up.json"
    circuit_factory.write_report(json_file, circuits, data, targets)
    print(f"\nCircuit data written to {json_file}")

    # Clean up the circuits
    circuit_factory.clean_up(circuits, data, targets)

    # write report after cleaning up the circuits
    json_file = "iris_binary_circuits_detailed_scores.json"
    circuit_factory.write_report(json_file, circuits, data, targets)
    print(f"\nCircuit data written to {json_file}")

    # which is the best circuit?
    best_circuit = None
    best_score = 0
    best_id = -1
    for i, circuit in enumerate(circuits):
        score_matrix = circuit.score(data, targets)
        max_accuracy_per_class = np.max(score_matrix, axis=0)
        if np.sum(max_accuracy_per_class) > best_score:
            best_score = np.sum(max_accuracy_per_class)
            best_id = i
            print('new best best_id', best_id)
            print('new best score', best_score)
            best_circuit = circuit
    print(f"Best Circuit ID: {best_id}")
    print(f"Best Circuit Score: {best_score}")
    print(f"Best Circuit Gates: {best_circuit.gates}")
    print(f"Best Circuit Designated Output Indices: {best_circuit.designated_output_indices}")

    # accuracy per class
    max_accuracy_per_class = np.max(score_matrix, axis=0)
    print(f"Max accuracy per class: {max_accuracy_per_class}")
    # best node index per class
    best_node_per_class = np.argmax(score_matrix, axis=0)
    print(f"Best node index per class: {best_node_per_class}")
    """
    Best Circuit Designated Output Indices: [23, 24, 25]
    Max accuracy per class: [0.86666667 0.86       0.96      ]
    Best node index per class: [13 14 15]
    """

    # calculate the average accuracy of all circuits
    average_accuracy_of_all_circuits = 0
    for i, circuit in enumerate(circuits):
        score_matrix = circuit.score(data, targets)
        max_accuracy_per_class = np.max(score_matrix, axis=0)
        average_accuracy_of_all_circuits += np.sum(max_accuracy_per_class)
    average_accuracy_of_all_circuits /= len(circuits)
    print(f"Average accuracy of all circuits #1: {average_accuracy_of_all_circuits}")

    circuit_count = len(circuits)
    print(f"Number of circuits: {circuit_count}")
    number_of_circuits_to_remove = circuit_count // 4 # remove 1/4 of the circuits
    keep_at_least = 50
    keeping = circuit_count - number_of_circuits_to_remove
    print(f"Number of circuits to keep: {keeping}")
    if keeping < keep_at_least:
        number_of_circuits_to_remove = keep_at_least - keeping

    print(f"Number of circuits to remove: {number_of_circuits_to_remove}")
    circuits = circuit_factory.remove_worst_circuits(circuits, data, targets, num_to_remove=number_of_circuits_to_remove)

    # calculate the average accuracy of all circuits
    average_accuracy_of_all_circuits = 0
    for i, circuit in enumerate(circuits):
        score_matrix = circuit.score(data, targets)
        max_accuracy_per_class = np.max(score_matrix, axis=0)
        average_accuracy_of_all_circuits += np.sum(max_accuracy_per_class)
    average_accuracy_of_all_circuits /= len(circuits)
    print(f"Average accuracy of all circuits #2: {average_accuracy_of_all_circuits}")



if __name__ == "__main__":
    # simple_test()
    # digits_test()
    iris_binary_test()

""" iris_binary_test output:
Best Circuit ID: 318
Best Circuit Score: 2.82
Best Circuit Gates: [('OR', 0, 7), ('NOT', 7), ('NAND', 12, 13), ('XNOR', 11, 12), ('NOT', 0), ('XOR', 14, 12), ('XOR', 4, 5), ('XNOR', 17, 4), ('NOT', 5), ('XNOR', 21, 20), ('NOR', 4, 3), ('NOR', 26, 6), ('XNOR', 27, 16), ('NAND', 27, 17), ('AND', 15, 17), ('XNOR', 8, 24), ('NAND', 8, 27), ('OR', 13, 12), ('XNOR', 0, 19), ('XOR', 23, 27), ('NOR', 10, 2), ('NAND', 25, 2), ('OR', 34, 22), ('AND', 6, 11), ('AND', 16, 13), ('OR', 33, 17), ('NOR', 35, 5), ('NOT', 33), ('XOR', 19, 27), ('OR', 6, 9), ('OR', 11, 7), ('NOT', 24), ('XOR', 15, 29), ('NOR', 43, 5), ('NAND', 48, 45), ('NOT', 19), ('NOR', 45, 46), ('AND', 50, 24), ('OR', 40, 33), ('NOT', 5), ('XOR', 26, 46), ('OR', 37, 22), ('OR', 47, 52), ('NOR', 51, 40), ('XOR', 23, 36), ('OR', 10, 4), ('NOR', 15, 31), ('NOR', 48, 12), ('NOT', 26), ('XNOR', 52, 24)]
Best Circuit Designated Output Indices: [63, 64, 65]
Max accuracy per class: [0.86666667 0.86       0.96      ]
Best node index per class: [13 14 15]
Average accuracy of all circuits #1: 2.697453333333337
Number of circuits: 500
Number of circuits to remove: 125
Average accuracy of all circuits #2: 2.693724444444422

we see that the accuracy dropped a bit, but not much.
we should find the median accuracy of the circuits and remove the worst ones.
we should also remove duplicates.
we should also remove circuits that are too similar to each other.
"""
