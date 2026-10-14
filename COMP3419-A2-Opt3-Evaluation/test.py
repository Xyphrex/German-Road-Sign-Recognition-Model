import os
from tensorflow.keras.models import load_model
from testloader import GTSRB_Test_Loader
from evaluation import evaluate

if __name__ == '__main__':

    model = load_model('../road_sign_model_old.keras')


    testloader = GTSRB_Test_Loader(TEST_PATH='../GTSRB Test/Final_Test/Images', TEST_GT_PATH='./GTSRB_Test_GT.csv')

    testing_accuracy = evaluate(model, testloader)
    print('Testing finished, accuracy: {:.3f}'.format(testing_accuracy))
