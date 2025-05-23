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
    json_file = "reports/iris_binary_circuits_detailed_scores_before_clean_up.json"
    circuit_factory.write_report(json_file, circuits, data, targets)
    print(f"\nCircuit data written to {json_file}")

    # Clean up the circuits
    circuit_factory.clean_up(circuits, data, targets)

    # write report after cleaning up the circuits
    json_file = "reports/iris_binary_circuits_detailed_scores.json"
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
    score_per_class = average_accuracy_of_all_circuits / num_target_classes
    print(f"Average score per class: {score_per_class}")

    # html report
    count = len(circuits)
    circuit_factory.write_report_html(f"reports/iris_binary_circuits_{count}_detailed_scores.html", circuits, data, targets)

    # remove the worst circuits
    circuit_count = len(circuits)
    print(f"Number of circuits: {circuit_count}")
    number_of_circuits_to_remove = circuit_count // 10 # remove bottom tenth of the circuits
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
    score_per_class = average_accuracy_of_all_circuits / num_target_classes
    print(f"Average score per class: {score_per_class}")

    # html report
    count = len(circuits)
    circuit_factory.write_report_html(f"reports/iris_binary_circuits_{count}_detailed_scores.html", circuits, data, targets)



if __name__ == "__main__":
    # simple_test()
    # digits_test()
    iris_binary_test()

""" iris_binary_test output:
Best Circuit ID: 40
Best Circuit Score: 2.8533333333333335
Best Circuit Gates: [('NOR', 8, 7), ('OR', 4, 7), ('AND', 8, 7), ('XOR', 13, 12), ('OR', 4, 19), ('NOR', 0, 5), ('NOR', 15, 4), ('XNOR', 16, 17), ('XNOR', 10, 15), ('NAND', 8, 17), ('XOR', 1, 12), ('AND', 22, 5), ('AND', 6, 26), ('NAND', 15, 26), ('XNOR', 3, 14), ('XNOR', 10, 8), ('NAND', 23, 16), ('NAND', 1, 26), ('AND', 22, 12), ('XOR', 34, 11), ('NAND', 22, 20), ('XNOR', 32, 31), ('XNOR', 28, 5), ('NOR', 18, 29), ('XNOR', 9, 10), ('XNOR', 21, 26), ('XNOR', 39, 36), ('NOR', 4, 5), ('OR', 21, 14), ('NAND', 20, 41), ('NAND', 45, 17), ('XOR', 41, 25), ('NAND', 28, 23), ('XNOR', 37, 33), ('NOR', 26, 10), ('OR', 1, 8), ('XNOR', 15, 42), ('NAND', 47, 44), ('XNOR', 5, 13), ('OR', 25, 6), ('AND', 4, 30), ('XOR', 3, 1), ('NOR', 33, 48), ('AND', 6, 16), ('XOR', 9, 5), ('XNOR', 59, 56), ('NOR', 19, 14), ('OR', 47, 18), ('NOR', 16, 21), ('AND', 4, 45)]
Best Circuit Designated Output Indices: [63, 64, 65]
Max accuracy per class: [0.87333333 0.86       0.96      ]
Best node index per class: [52 14 15]
Average accuracy of all circuits #1: 2.6970266666666673
Average score per class: 0.8990088888888891
HTML report written to reports/iris_binary_circuits_500_detailed_scores.html
Number of circuits: 500
Number of circuits to keep: 450
Number of circuits to remove: 50
Average accuracy of all circuits #2: 2.6955259259259186
Average score per class: 0.8985086419753062
HTML report written to reports/iris_binary_circuits_450_detailed_scores.html

we see that the accuracy dropped a bit, but not much.
we should find the median accuracy of the circuits and remove the worst ones.
we should also remove duplicates.
we should also remove circuits that are too similar to each other.
"""
