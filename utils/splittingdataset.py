from sklearn.model_selection import train_test_split

def train_test_val_split(dataset, test_size=0.3, val_size=0.1, random_state = 2023):
    # Split data into train and test sets
    train_data_set, test_data_set = train_test_split(dataset, test_size=test_size, random_state=2023)

    # Split test data into test and validation sets
    val_data_set, test_dataset  = train_test_split(test_data_set, test_size=val_size/(1-test_size), random_state=2023)

    return train_data_set,  val_data_set, test_dataset
