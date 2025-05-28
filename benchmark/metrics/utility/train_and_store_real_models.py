from benchmark.metrics.utility.train_model_classification import create_categorical_trained_model
from benchmark.metrics.utility.train_model_regression import create_regression_trained_model
from benchmark.metrics.utility.train_model_classification_mv import create_categorical_trained_model_market_value
from benchmark.utils import load_real_data_train, load_real_data_test

# run from root directory using "python3 -m benchmark.metrics.utility.train_and_store_real_models"
real_data_train = load_real_data_train()
real_data_test = load_real_data_test()
#create_categorical_trained_model(real_data, real=True, name="real") #TODO comment in if needed
# create_categorical_trained_model_market_value(real_data_train, real_data_test, real=True, name="real")
create_regression_trained_model(real_data_train, real_data_test, real=True, name="real_with_split")