from numpy import *
import csv

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        
class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # Сигмоид
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Градиент для сигмоида
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    # Обучение сети
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2, classif = self.get_outputs(training_set_inputs)
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # Определение значий на выход
    def get_outputs(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        classif = []
        for i in range(output_from_layer2.shape[0]):
            classif.append(where(output_from_layer2[i] == max(output_from_layer2[i]))[0] + 1)
        return output_from_layer1, output_from_layer2, classif

    # Вывод весов
    def print_weights(self):
        print ("Слой 1:")
        print (self.layer1.synaptic_weights)
        print ("Слой 2:")
        print (self.layer2.synaptic_weights)
    # Вывод матрицы неточностей
    def get_confusion_matrix(self, correct_outputs, predicted_outputs):
        conf_matr = zeros((3,3))
        for i in range(len(correct_outputs)):
            conf_matr[int(correct_outputs[i]) - 1, int(predicted_outputs[i]) - 1] = conf_matr[int(correct_outputs[i]) - 1, int(predicted_outputs[i]) - 1] + 1
        return conf_matr
    def get_precision(self, correct_outputs, predicted_outputs, cl):
        matr = self.get_confusion_matrix(correct_outputs, predicted_outputs)
        return matr[cl, cl] / sum(matr[cl,:])
    def get_recall(self, correct_outputs, predicted_outputs, cl):
        matr = self.get_confusion_matrix(correct_outputs, predicted_outputs)
        return matr[c, c] / sum(matr[:,c])
    def get_Fmeasure(self, correct_outputs, predicted_outputs, cl):
        matr = self.get_confusion_matrix(correct_outputs, predicted_outputs)
        return 2 * precision * recall / (precision + recall)

if __name__ == "__main__":
    # Подготовка датасета
    dataset = []
    with open('iris_full.csv', 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            for i in range(5):
                row[i] = float(row[i])
            dataset.append(row)
    dataset = array(dataset)
    dataset = dataset[:,1:]
    # Нормализация данных
    for j in range(dataset.shape[1] - 1):
        min_j = min(dataset[:, j]).astype(float)
        max_j = max(dataset[:, j]).astype(float)
        for i in range(dataset.shape[0]):
            dataset[i, j] = float(dataset[i,j])
            dataset[i, j] = (dataset[i,j].astype(float)-min_j) / (max_j - min_j)
    # Переход от названий видов к номерам классов
    for i in range(dataset.shape[0]):
        if(dataset[i, 4] == 'setosa'):
            dataset[i,4] = 1
        elif(dataset[i,4] == 'versicolor'):
            dataset[i, 4] = 2
        else:
             dataset[i,4] = 3
    # Создание обучающей и тестовой выборок
    training_inputs = concatenate((dataset[:15, :4], dataset[51:66,:4],dataset[101:116, :4]), axis = 0).astype(float)
    tr_outputs = concatenate((dataset[:15, 4:], dataset[51:66,4:],dataset[101:116, 4:]), axis = 0).astype(float)
    if(tr_outputs[0, 0] == 1):
        training_outputs = array([[1,0,0]])
    elif(tr_outputs[0, 0] == 2):
        training_outputs = array([[0,1,0]])
    else:
        training_outputs = array([[0,0,1]])
    for i in range(1, tr_outputs.shape[0]):
        if(tr_outputs[i, 0] == 1):
            training_outputs = concatenate((training_outputs, array([[1,0,0]])), axis = 0)
        elif(tr_outputs[i, 0] == 2):
            training_outputs = concatenate((training_outputs, array([[0,1,0]])), axis = 0)
        else:
            training_outputs = concatenate((training_outputs, array([[0,0, 1]])), axis = 0)

    testing_inputs = concatenate((dataset[15:51, :4], dataset[66:101,:4],dataset[116:, :4]), axis = 0).astype(float)
    cor_classif = concatenate((dataset[15:51, 4:], dataset[66:101,4:],dataset[116:, 4:]), axis = 0).astype(float)
    tes_outputs = concatenate((dataset[15:51, 4:], dataset[66:101,4:],dataset[116:, 4:]), axis = 0).astype(float)
    
    if(tes_outputs[0, 0] == 1):
        testing_outputs = array([[1,0,0]])
    elif(tes_outputs[0, 0] == 2):
        testing_outputs = array([[0,1,0]])
    else:
        testing_outputs = array([[0,0,1]])

    for i in range(1, tes_outputs.shape[0]):
        if(tes_outputs[i, 0] == 1):
            testing_outputs = concatenate((testing_outputs, array([[1,0,0]])), axis = 0)
        elif(tes_outputs[i, 0] == 2):
            testing_outputs = concatenate((testing_outputs, array([[0,1,0]])), axis = 0)
        else:
            testing_outputs = concatenate((testing_outputs, array([[0,0,1]])), axis = 0)
    
    random.seed(1)
    
    # Первый слой из 4 нейронов
    layer1 = NeuronLayer(4, training_inputs.shape[1])
    # Второй слой из 3 нейронов
    layer2 = NeuronLayer(3, 4)

    # Создание нейронной сети
    neural_network = NeuralNetwork(layer1, layer2)

    # Начальные веса
    print("Начальные веса")
    neural_network.print_weights()
    # Обучение сети
    neural_network.train(training_inputs, training_outputs, 60000)
    # Веса после обучения
    print("Веса после обучения")
    neural_network.print_weights()

    # Тестирование сети
    layer1_out, layer2_out, classif = neural_network.get_outputs(testing_inputs)
    
   

    cl = zeros(3)
    for i in range(cor_classif.shape[0]):
        cl[int(cor_classif[i]) - 1] = cl[int(cor_classif[i]) - 1] + 1
    print("Количество объектов в каждом классе", cl)
    
    matr = neural_network.get_confusion_matrix(cor_classif, classif)
    print("Матрица неточностей")
    print(matr)
    names = ("setosa", "versicolor", "virginica")
    for c in range(3):
        print("Класс №", c + 1, "(", names[c], ") :")
        precision = neural_network.get_precision(cor_classif, classif ,c)
        print("точность / precision:", precision)
        recall =  neural_network.get_recall(cor_classif, classif ,c)
        print("полнота / recall:", recall)
        Fmeasure = neural_network.get_Fmeasure(cor_classif, classif ,c)
        print("F-мера / F-measure", Fmeasure)
    
    
    
            
