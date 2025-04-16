# Digital Circuit Learning

This project explores the use of digital circuits for supervised learning tasks.
It includes functionality for creating, training, and evaluating circuits on different datasets.

## Overview

The project utilizes a `Circuit` class and a `CircuitFactory` class to generate and manage digital circuits.
These circuits are composed of basic logic gates such as NOT, AND, OR, NAND, NOR, XOR, and XNOR.
The circuits are trained and evaluated on datasets to perform classification tasks.

## Key Components

- `circuit_maker.py`: Contains the `Circuit` and `CircuitFactory` classes, which are responsible for creating and manipulating digital circuits.
- `main.py`: Contains the main execution logic, including examples of how to create and train circuits using different datasets.
- `datasets/`: Contains example datasets, such as the Iris binary dataset and the Digits dataset.

## Logic Gates

The following logic gates are used in the circuits:

| Gate        | Description |
| ----------- | ----------- |
| NOT(a)      | 1 - a       |
| AND(a, b)   | a & b       |
| OR(a, b)    | a \| b      |
| NAND(a, b)  | 1 - (a & b) |
| NOR(a, b)   | 1 - (a \| b)|
| XOR(a, b)   | a ^ b       |
| XNOR(a, b)  | 1 - (a ^ b) |

## Datasets

The project currently supports the following datasets:

- Iris Binary Dataset: A binary version of the Iris dataset, located in `datasets/iris_binary.csv`.
- Digits Dataset: A dataset of handwritten digits, located in `datasets/digits.csv`.

## Usage

The `main.py` file provides examples of how to use the `CircuitFactory` to create and train circuits. You can run the `main.py` file to execute these examples.

```bash
python main.py
```

The `main.py` script includes the following tests:

- Simple Test: A basic example of creating and training circuits with a randomly generated dataset.
- Digits Test: An example of creating and training circuits with the digits dataset.
- Iris Binary Test: An example of creating and training circuits with the iris binary dataset.

To run a specific test, uncomment the corresponding line in the `if __name__ == "__main__":` block in `main.py`. For example, to run the Iris binary test, uncomment the line `iris_binary_test()`.

## Output

The project generates JSON files containing detailed scores for each circuit. These files are stored in the project directory.

- `digits_circuits_detailed_scores.json`: Contains the scores for circuits trained on the digits dataset.
- `iris_binary_circuits_detailed_scores.json`: Contains the scores for circuits trained on the iris binary dataset.

## Example Circuit Data

The JSON files contain data for each circuit, including the gates used, the designated output indices, and the score matrix. Here's an example of the data format:

```json
{
    "circuit_id": 0,
    "gates": "[('XNOR', 5, 3), ('XOR', 35, 58), ('XOR', 11, 13), ('NOT', 59), ('NOT', 23)]",
    "designated_output_indices": [64, 65, 66, 67, 68],
        "score_matrix": [
            [
                0.8266666666666667,
                0.49333333333333335,
                0.4666666666666667
            ],
            //..
        ]
}
```
