import config
from data_handler import DataHandler
from model import RandomForestModel 

def run_multi_label_controller():
    # 1. Initialize the Components
    handler = DataHandler()
    df = handler.load_process_data()
    
    # 2. Initialize the Model
    model = RandomForestModel()
    
    # 3. Design Choice 1: Chained Evaluation Loop
    chained_targets = ['y2', 'y23', 'y234']
    
    print("--- Starting Chained Multi-Label Classification ---")
    
    for target in chained_targets:
        X_train, X_test, y_train, y_test = handler.get_splits(target)
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        model.print_results(y_test, predictions, target)

if __name__ == "__main__":
    run_multi_label_controller()