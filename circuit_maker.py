import numpy as np
import random
import json

# gates
class GateLogic:
    GATE_FUNCTIONS = {
        'AND': lambda a, b: a & b,
        'OR': lambda a, b: a | b,
        'NAND': lambda a, b: 1 - (a & b),
        'NOR': lambda a, b: 1 - (a | b),
        'XOR': lambda a, b: a ^ b,
        'XNOR': lambda a, b: 1 - (a ^ b)
    }
    GATE_TYPES = list(GATE_FUNCTIONS.keys())

class Circuit:
    """Circuit class.
    Each circuit has some inputs, outputs, and gates.
    Gates are added sequentially. When a gate is added, its inputs are randomly
    selected from the pool of available nodes (primary inputs + outputs of preceding gates)."""
    def __init__(self, num_inputs, num_target_classes, num_gates):
        self.num_inputs = num_inputs
        self.num_target_classes = num_target_classes
        self.num_gates = num_gates
        self.gates = [] # List to store (gate_type, input_idx1, [input_idx2])

        # Total number of nodes available for input connections
        # starts with primary inputs
        num_available_nodes = num_inputs

        for i in range(num_gates):
            # Randomly select a gate type
            gate_type = random.choice(GateLogic.GATE_TYPES)
            #arity = GateLogic.GATE_ARITY[gate_type]

            # Ensure we have enough available nodes for the gate's inputs
            if num_available_nodes < 2:
                raise ValueError("Cannot create gate, not enough input nodes available.")

            # Randomly select input node indices for the current gate.
            # Inputs are chosen from the pool of all available nodes created so far:
            # - Indices 0 to num_inputs-1 represent the primary circuit inputs.
            # - Indices num_inputs to num_available_nodes-1 represent the outputs of previously added gates.
            inputs = random.sample(range(num_available_nodes), 2)
            self.gates.append((gate_type, inputs[0], inputs[1]))

            # The output of this gate becomes a new available node
            num_available_nodes += 1

        # Total number of nodes (wires) in the circuit = inputs + gates
        self.total_nodes = self.num_inputs + self.num_gates

        # Define circuit *designated* output indices (e.g., outputs of the last num_outputs gates)
        # These might still be useful for interpretation or specific selection later
        # Node indices are num_inputs + gate_index
        # Ensure we don't try to select outputs from before the first gate
        first_gate_output_idx = num_inputs
        last_gate_output_idx = self.total_nodes -1
        # Make sure num_outputs doesn't exceed the number of gates
        actual_num_outputs = min(self.num_target_classes, self.num_gates)

        if self.num_gates > 0 :
            self.designated_output_indices = list(range(last_gate_output_idx - actual_num_outputs + 1 , last_gate_output_idx + 1 ))
            # Check if any index is invalid (this logic might need adjustment if num_gates < num_outputs)
            if not self.designated_output_indices or self.designated_output_indices[0] < first_gate_output_idx:
                 # Handle cases where num_gates < num_outputs gracefully, maybe output all available gates
                 print(f"Warning: Requested {self.num_target_classes} outputs, but only {self.num_gates} gates exist. Using outputs of gates {list(range(num_inputs, self.total_nodes))}")
                 self.designated_output_indices = list(range(first_gate_output_idx, self.total_nodes))
        else:
            self.designated_output_indices = [] # No gates, no gate outputs


    def evaluate(self, inputs):
        """Evaluate the circuit and return the values of ALL nodes (inputs + gate outputs)."""
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, but got {len(inputs)}")
        if not all(i in [0, 1] for i in inputs):
              raise ValueError("Inputs must be binary (0 or 1)")

        # Buffer to hold values of all nodes (inputs + gate outputs)
        node_values = np.zeros(self.num_inputs + len(self.gates), dtype=int)
        node_values[:self.num_inputs] = inputs

        # Evaluate gates sequentially
        for i, gate_info in enumerate(self.gates):
            gate_type = gate_info[0]
            gate_func = GateLogic.GATE_FUNCTIONS[gate_type]
            # The index where the output of the i-th gate will be stored
            gate_output_index = self.num_inputs + i
            input_idx1 = gate_info[1]
            input_idx2 = gate_info[2]
            if input_idx1 < len(node_values) and input_idx2 < len(node_values):
                input_val1 = node_values[input_idx1]
                input_val2 = node_values[input_idx2]
                gate_output = int(gate_func(input_val1, input_val2))
                node_values[gate_output_index] = gate_output

        # Return the values of ALL nodes
        return node_values

    def score(self, data, targets):
        """
        Scores the circuit by comparing each node's output against each target class output.

        Args:
            data (list of lists or np.ndarray): Input data samples.
            targets (list of lists or np.ndarray): One-hot encoded target values.

        Returns:
            np.ndarray: A 2D array where element [i, j] is the accuracy score
                        of node 'i' predicting target class 'j'.
                        Shape: (total_nodes, num_target_classes)
        """
        if len(data) == 0:
            return np.zeros((self.total_nodes, self.num_target_classes)) # Return zeros if no data
        if len(data) != len(targets):
            raise ValueError("Number of data points must match number of targets.")
        if not isinstance(targets, np.ndarray):
             targets = np.array(targets) # Ensure targets is a numpy array for easier slicing
        if targets.shape[1] != self.num_target_classes:
             raise ValueError(f"Target shape mismatch. Expected {self.num_target_classes} classes, but got {targets.shape[1]}.")


        # Initialize a matrix to store the count of correct predictions
        # Rows: Nodes (wires), Columns: Target Classes
        correct_predictions_matrix = np.zeros((self.total_nodes, self.num_target_classes), dtype=int)

        num_samples = len(data)
        for i in range(num_samples):
            inputs = data[i]
            target_vector = targets[i]

            # Get the values of all nodes for this input
            all_node_values = self.evaluate(inputs) # Shape: (total_nodes,)

            # Compare each node's output to each target class value
            for node_idx in range(self.total_nodes):
                node_output = all_node_values[node_idx]
                for class_idx in range(self.num_target_classes):
                    target_class_value = target_vector[class_idx]
                    if node_output == target_class_value:
                        correct_predictions_matrix[node_idx, class_idx] += 1

        # Calculate accuracy for each (node, target_class) pair
        accuracy_matrix = correct_predictions_matrix / num_samples

        return accuracy_matrix

class CircuitFactory:
    """Circuit factory to build multiple random circuits."""
    def __init__(self, num_inputs, num_target_classes, num_gates):
        self.num_inputs = num_inputs
        self.num_gates = num_gates
        self.num_target_classes = num_target_classes

    def build_circuits(self, num_circuits):
        """Builds a list of random circuits."""
        circuits = []
        for _ in range(num_circuits):
            # Pass num_outputs as the number of target classes
            circuit = Circuit(self.num_inputs, self.num_target_classes, self.num_gates)
            circuits.append(circuit)
        return circuits

    def update(self, circuits, data, targets):
        """
        Scores each circuit and sets the designated outputs to match their best performing output.
        """
        for circuit in circuits:
            # Score the circuit
            score_matrix = circuit.score(data, targets)

            # For each target class, find the node (wire) with the highest accuracy
            best_performing_nodes = np.argmax(score_matrix, axis=0)
            print("update circuit designated outputs")
            print(f"  Circuit {circuit}: Best performing nodes: {best_performing_nodes}")

            # Set the designated outputs to match the best performing nodes
            circuit.output_indices = best_performing_nodes.tolist()

    def write_report(self, filename=None, circuits=None, data=None, targets=None):
        if not filename:
            random_8 = str(random.randint(0, 99999999))
            filename = 'circuit_report_' + random_8 + '.json'
        
        # Score Circuits
        circuit_data = []
        for i, circuit in enumerate(circuits):
            print(f"Scoring Circuit {i}...")
            score_matrix = circuit.score(data, targets) # Pass binarized data
            circuit_data.append({
                'circuit_id': i,
                'gates': str(circuit.gates),
                'designated_output_indices': circuit.designated_output_indices,
                'score_matrix': score_matrix.tolist() # Convert for JSON
            })
            print(f"  Circuit {i}: Score Matrix Shape = {score_matrix.shape}")
            # You could print summary stats, e.g., max accuracy for each class
            max_acc_per_class = np.max(score_matrix, axis=0)
            best_node_per_class = np.argmax(score_matrix, axis=0)
            print(f"  Max accuracy per class: {np.round(max_acc_per_class, 3)}")
            print(f"  Best node index per class: {best_node_per_class}")


        # Write circuit data to JSON
        try:
            with open(filename, 'w') as outfile:
                json.dump(circuit_data, outfile, indent=4) # Keep indent for readability if needed, remove for smaller files
            print(f"\nCircuit data written to {filename}")
        except IOError:
            print("I/O error")
        print("-" * 20)

    def write_report_html(self, filename="circuit_report.html", circuits=None, data=None, targets=None):
        """
        Writes an HTML report containing a table for each circuit,
        showing the output wires and their scores for each target class.
        """
        if not isinstance(circuits, list) or not circuits:
            print("Error write_report_html: No circuits provided.")
            return
        if not isinstance(data, np.ndarray) or data.size == 0:
            print("Error write_report_html: No data provided.")
            return
        if not targets.size == 0 and not isinstance(targets, np.ndarray):
            print("Error write_report_html: targets must be a numpy array.")
            return
        if len(data) != len(targets):
            raise ValueError("Number of data points must match number of targets.")

        num_target_classes = circuits[0].num_target_classes if circuits else 0
        total_nodes = circuits[0].total_nodes if circuits else 0

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Circuit Report</title>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>Circuit Report</h1>
        """

        for i, circuit in enumerate(circuits):
            score_matrix = circuit.score(data, targets)
 
            # Find the best score for each class
            best_scores = np.max(score_matrix[circuit.num_inputs:circuit.total_nodes], axis=0)
            best_nodes = np.argmax(score_matrix[circuit.num_inputs:circuit.total_nodes], axis=0) + circuit.num_inputs
 
            html_content += f"""
            <h2>Circuit {i}</h2>
            <table>
                <tr>
                    <th>Output Wire</th>
        """
            for class_idx in range(num_target_classes):
                html_content += f"<th>Class {class_idx}</th>"
            html_content += "</tr>"
 
            for node_idx in range(circuit.num_inputs, circuit.total_nodes):  # Iterate over output wires
                html_content += "<tr>"
                output_wire_style = ""
                html_content += f"<td>"
                for class_idx in range(num_target_classes):
                    if node_idx == best_nodes[class_idx]:
                        output_wire_style = '<b><span style="color: green;">'
                        break
                if output_wire_style:
                    html_content += f"{output_wire_style}{node_idx}</span></b>"
                else:
                    html_content += f"{node_idx}"
                html_content += f"</td>"  # Output wire number
 
                for class_idx in range(num_target_classes):
                    score = score_matrix[node_idx, class_idx]
                    score_style = ""
                    if node_idx == best_nodes[class_idx]:
                        score_style = '<b><span style="color: green;">'
                    if score_style:
                        html_content += f"<td>{score_style}{score:.3f}</span></b></td>"  # Score for this class
                    else:
                        html_content += f"<td>{score:.3f}</td>"  # Score for this class
                html_content += "</tr>"
 
            html_content += "</table><br><br>"

        html_content += """
        </body>
        </html>
        """

        try:
            with open(filename, 'w') as outfile:
                outfile.write(html_content)
            print(f"HTML report written to {filename}")
        except IOError:
            print("I/O error")

    def clean_up(self, circuits, data, targets):
        """
        Removes the worst-performing gates from a circuit, focusing on the gates
        that do not contribute to the best output wires.

        Args:
            circuits (list): A list of Circuit objects.
            data (list of lists or np.ndarray): Input data samples.
            targets (list of lists or np.ndarray): One-hot encoded target values.
        """
        remove_extra = False
        for circuit in circuits:
            # Score the circuit
            score_matrix = circuit.score(data, targets)

            # Identify the best output wires (nodes) for each target class
            best_output_indices = np.argmax(score_matrix, axis=0)
            # with 3 outputs we have something like
            # array([64, 14, 15], dtype=int64)

            # remove all gates greater than the max(best_output_indices)
            gates_to_remove = [i for i in range(len(circuit.gates)) if i > max(best_output_indices)]

            # Remove the higher useless gates
            # Removing gates in reverse order to avoid index shifting issues
            gates_to_remove.sort(reverse=True)
            for gate_index in gates_to_remove:
                del circuit.gates[gate_index]

            if remove_extra:
                # Identify gates with unused outputs
                used_nodes = set(range(circuit.num_inputs))  # Start with input nodes
                for i, gate in enumerate(circuit.gates):
                    used_nodes.add(gate[1])
                    used_nodes.add(gate[2])

                for i in range(len(circuit.gates)):
                    gate_output_idx = circuit.num_inputs + i
                    if gate_output_idx not in used_nodes and i not in gates_to_remove:
                        gates_to_remove.append(i)

                # Determine the gates that contribute to the best output wires
                relevant_gates = set()
                for output_idx in best_output_indices:
                    # Trace back the gates that contribute to this output
                    self._trace_gates(circuit, output_idx, relevant_gates)

                # Identify the gates to remove (gates that are not relevant)
                gates_to_remove = [i for i in range(len(circuit.gates)) if i not in relevant_gates]

                # Remove the irrelevant gates
                # Removing gates in reverse order to avoid index shifting issues
                gates_to_remove.sort(reverse=True)
                for gate_index in gates_to_remove:
                    del circuit.gates[gate_index]

            circuit.num_gates = len(circuit.gates)
            circuit.total_nodes = circuit.num_inputs + circuit.num_gates
        


    def remove_worst_circuits(self, circuits, data, targets, num_to_remove):
        """Removes the worst-performing circuits from a list of circuits.

        Args:
            circuits (list): A list of Circuit objects.
            data (list of lists or np.ndarray): Input data samples.
            targets (list of lists or np.ndarray): One-hot encoded target values.
            num_to_remove (int): The number of circuits to remove.
        """
        if num_to_remove <= 0 or not circuits:
            return circuits

        # Score each circuit
        circuit_scores = []
        for circuit in circuits:
            score_matrix = circuit.score(data, targets)
            # Calculate the average score for each circuit
            avg_score = np.mean(score_matrix)
            circuit_scores.append(avg_score)

        # Sort the circuits based on their average scores
        sorted_circuits = sorted(zip(circuits, circuit_scores), key=lambda x: x[1])

        # Remove the worst-performing circuits from the list
        circuits_to_keep = [circuit for circuit, score in sorted_circuits[num_to_remove:]]

        return circuits_to_keep

    def _trace_gates(self, circuit, node_idx, relevant_gates):
        """
        Recursively traces back the gates that contribute to a given node.

        Args:
            circuit (Circuit): The circuit object.
            node_idx (int): The index of the node to trace back from.
            relevant_gates (set): A set to store the indices of relevant gates.
        """
        if node_idx < circuit.num_inputs:
            # This is a primary input, so stop tracing
            return

        # Determine the gate index that produces this node
        gate_index = node_idx - circuit.num_inputs

        if gate_index < 0 or gate_index >= len(circuit.gates):
            # This should not happen, but handle it gracefully
            return

        relevant_gates.add(gate_index)

        # Recursively trace back the inputs to this gate
        # Check if gate_index is within the valid range before accessing circuit.gates[gate_index]
        if 0 <= gate_index < len(circuit.gates):
            gate = circuit.gates[gate_index]
            gate_type = gate[0]
            input_idx1 = gate[1]
            input_idx2 = gate[2]
            self._trace_gates(circuit, input_idx1, relevant_gates)
            self._trace_gates(circuit, input_idx2, relevant_gates)

    def _update_gate_indices(self, circuit):
       """
       Updates the gate indices in the circuit after removing gates.
       """
       gates_removed_count = 0
       for i in range(len(circuit.gates)):
           gate = circuit.gates[i]
           gate_type = gate[0]
           input_idx1 = gate[1]
           input_idx2 = gate[2]
           circuit.gates[i] = (gate_type, input_idx1 - gates_removed_count, input_idx2 - gates_removed_count)
           gates_removed_count += 1