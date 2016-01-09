__author__ = 'greg'
import loader
import neural_network

training_data, validation_data, test_data = loader.load_data_wrapper()
net = neural_network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)