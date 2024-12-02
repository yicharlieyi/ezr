
import os
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK, space_eval, rand, anneal
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler 

def list_datasets(base_path):
    return [os.path.join(base_path, file) for file in os.listdir(base_path) if file.endswith('.csv')]

datasets = list_datasets('C:/Users/User/CSC591/ezr/data/optimize/config/')

def load_and_preprocess_data(dataset_path):
    # Load dataset
    data = pd.read_csv(dataset_path)
    
    # Shuffle data
    data = shuffle(data, random_state=42)
    
    # Assuming the last column is the target variable
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Splitting data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    
    return train_x, train_y, val_x, val_y

def build_and_train_model(params, train_x, train_y):
    model = Sequential()
    for _ in range(int(params['num_layers'])):
        model.add(Dense(params['units_per_layer'], activation='relu'))
        model.add(Dropout(params['dropout_rate']))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                  loss='binary_crossentropy',  # or change according to your problem
                  metrics=['accuracy'])
    
    model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=0)
    return model

def evaluate_model(model, val_x, val_y):
    # Evaluate the model
    loss, accuracy = model.evaluate(val_x, val_y, verbose=0)
    return loss, accuracy  # Return both loss and accuracy


results = {}

space = {
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'units_per_layer': hp.choice('units_per_layer', [32, 64, 128, 256]),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
}

for dataset in datasets:
    def objective(params):
        train_x, train_y, val_x, val_y = load_and_preprocess_data(dataset)
        model = build_and_train_model(params, train_x, train_y)
        val_loss, val_accuracy = evaluate_model(model, val_x, val_y)
        combined_metric = val_loss - val_accuracy * 0.01
        return {'loss': combined_metric, 'status': STATUS_OK}

# Configure the algorithm to use and how many evaluations
algo = rand.suggest  # Random Search
max_evals = 50

trials = Trials()
best = fmin(fn=objective, space=space, algo=algo, max_evals=max_evals, trials=trials)

print("Best parameters found: ", space_eval(space, best))
