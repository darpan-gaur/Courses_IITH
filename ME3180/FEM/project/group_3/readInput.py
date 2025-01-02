import numpy as np

def readInputData(filename):
    """
    Read input data from a text file.

    By:
        Aaryan
        CO21BTECH11001
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = {}

    # Parse material properties
    data['Material_Props'] = {}
    for i in range(len(lines)):
        if i < len(lines) - 1 and 'Material_Props:' in lines[i]:
            data['Material_Props']['Youngs_modulus'] = float(lines[i+1].split(':')[1])
            data['Material_Props']['Moment_of_inertia'] = float(lines[i+2].split(':')[1])

    # Parse number of nodes
    for i in range(len(lines)):
        if i < len(lines) and 'No._nodes:' in lines[i]:
            data['No._nodes'] = int(lines[i].split(':')[1])

    # Parse nodal coordinates
    data['Nodal_coords'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'Nodal_coords:' in lines[i]:
            j = i + 1
            while j < len(lines) and lines[j].strip():
                data['Nodal_coords'].append(float(lines[j]))
                j += 1

    data['Nodal_coords'] = np.array(data['Nodal_coords'])

    # Parse element connectivity
    data['Element_connectivity'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'Element_connectivity:' in lines[i]:
            j = i + 1
            while j < len(lines) and lines[j].strip():
                data['Element_connectivity'].append(list(map(int, lines[j].split())))
                j += 1

    data['Element_connectivity'] = np.array(data['Element_connectivity'])

    # Parse prescribed DOFs
    data['Prescribed_DOFs'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'No._nodes_with_prescribed_DOFs:' in lines[i]:
            num_dofs = int(lines[i].split(':')[1])
            j = i + 2  # Skip "Node_#, DOF#, Value"
            for _ in range(num_dofs):
                if j < len(lines):
                    node_dof_value = list(map(float, lines[j].split()))
                    data['Prescribed_DOFs'].append(node_dof_value)
                    j += 1

    data['Prescribed_DOFs'] = np.array(data['Prescribed_DOFs'])

    # Parse spring stiffness
    data['Spring_stiffness'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'No._nodes_with_spring_stiffness:' in lines[i]:
            num_springs = int(lines[i].split(':')[1])
            j = i + 2
            for _ in range(num_springs):
                if j < len(lines):
                    node_spring_value = list(map(float, lines[j].split()))
                    data['Spring_stiffness'].append(node_spring_value)
                    j += 1

    data['Spring_stiffness'] = np.array(data['Spring_stiffness'])

    # Parse prescribed loads
    data['Prescribed_loads'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'No._nodes_with_prescribed_loads:' in lines[i]:
            num_loads = int(lines[i].split(':')[1])
            j = i + 2  # Skip "Node_#, DOF#, Traction_components"
            for _ in range(num_loads):
                if j < len(lines):
                    node_load_value = list(map(float, lines[j].split()))
                    data['Prescribed_loads'].append(node_load_value)
                    j += 1

    data['Prescribed_loads'] = np.array(data['Prescribed_loads'])

    # Parse body forces
    data['Body_forces'] = []
    for i in range(len(lines)):
        if i < len(lines) and 'No._prescribed_bodyforces:' in lines[i]:
            num_bodyforces = int(lines[i].split(':')[1])
            j = i + 1
            for _ in range(num_bodyforces):
                if j+1 < len(lines):
                    bodyforce_type = lines[j].split(':')[1].strip()
                    data['Body_forces'].append({'type': bodyforce_type, 'start_end_coords': [], 'val': []})
                    k = j + 2  # Skip "Coordinate, Value:"
                    while k < len(lines) and lines[k].strip() and lines[k].split()[0] != 'type:':
                        coord, val = tuple(map(float, lines[k].split(',')))
                        data['Body_forces'][-1]['start_end_coords'].append(coord)
                        data['Body_forces'][-1]['val'].append(val)
                        k += 1
                        
                    j = k

    return data