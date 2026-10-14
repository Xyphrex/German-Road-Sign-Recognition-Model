import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate(model, testloader):
    total_count = 0
    correct_count = 0
    class_count = np.zeros(43)
    class_labels = np.arange(43)
    class_sum = np.array([60, 720, 750, 450, 660, 630, 150, 450, 450, 480, 660, 420, 690, 720, 270, 210, 150, 360, 390, 
                          60, 90, 90, 120, 150, 90, 480, 180, 60, 150, 90, 150, 270, 60, 210, 120, 390, 120, 60, 690, 90, 90, 60, 90])
    evaluate_results = []


    for img, label, filename in tqdm(testloader):
        output = model.predict(img)
        predict = np.argmax(output, axis=1)
        
        batch_size = len(label) if isinstance(label, (np.ndarray, list)) else 1

        total_count += batch_size
        correct_count += np.sum(predict == label)

        if isinstance(label, (np.ndarray, list)):
            for i in range(batch_size):
                if label[i] == predict[i]:
                    class_count[label[i]] += 1
                else:
                    evaluate_results.append([label[i], filename[i]])
        else:
            if label == predict:
                class_count[label] += 1
            else:
                evaluate_results.append([label, filename])

    testing_accuracy = correct_count / total_count
    print("Test Accuracy: {:.4f}".format(testing_accuracy))

    class_accuracy = class_count / class_sum * 100
    print("Each class accuracy:\n", class_accuracy)

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_accuracy, color='skyblue')

    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for Each Class')

    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.show()

    return testing_accuracy
