
PROXIMITY_VECTOR_LABELS = {
    0: [0,0,0], 
    1: [1,0,0], 
    2: [0,1,0], 
    3: [0,0,1], 
    4: [1,0,1], 
    5: [1,1,0], 
    6: [0,1,1], 
    7: [1,1,1]
}

PROXIMITY_VECTOR_LABELS_FOR_TRAINING = {
    0: [1,0,0,0],
    1: [0,1,0,0],
    2: [0,0,1,0],
    3: [0,0,0,1],
    4: [0,1,0,1],
    5: [0,1,1,0],
    6: [0,0,1,1],
    7: [0,1,1,1],
}

PROXIMITY_CLASS_NAMES = ['sin_anomalias', 'condensacion', 'nodulos', 'quistes']