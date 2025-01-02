To run the code, command:
python main.py <input_file>


Example:
python main.py input_beam.txt


Input file format:
Material_Props:
    Young's_modulus:   <value>
    Moment_of_inertia: <value>

No._nodes: <number_of_nodes>
Nodal_coords:
    <coord_1>
    <coord_2>
    ...

No._elements: <number_of_elements>
No._nodes_per_elements: 2
Element_connectivity:
    <node_1> <node_2>
    <node_2> <node_3>
    ...


No._nodes_with_prescribed_DOFs: <number_of_nodes>
Node_#, DOF#, Value:
    <node_#> <DOF_#> <value>
    ...

No._nodes_with_spring_stiffness: <number_of_nodes>
Node_#, Stiffness (k):
    <node_#> <stiffness>

No._nodes_with_spring_stiffness: <number_of_nodes>
Node_#, Stiffness (k):
    <node_#> <stiffness>

No._prescribed_bodyforces: <number_of_forces>
    type: <linear_or_other_type>
    Coordinate, Value:
        <start_coord>, <start_value>
        <end_coord>, <end_value>
        ...
    ...

