from work.models.NeuralNetworks.NeuralNetworks import *


class DNN(NeuralNetwork):
    def __init__(self, name, model_):
        NeuralNetwork.__init__(self, name, model_)

    def create_network(self, input_shape=None):
        """ 
        Helper method to construct the network using the pre-defined
        parameters. It is used at the begining to create the model but also if
        the model has to be retrained from scratch to include validation data.
        """
        model = tf.keras.Sequential()
        
        # Add layers on top of each other. If the batch norm has to be used,
        # then batch normalization layers will be added between each layers.
        try:
            self.neurons_per_layer[0]
        except:
            self.neurons_per_layer = (self.neurons_per_layer, )
            
        for i, neuron in enumerate(self.neurons_per_layer):
            try:
                activation = self.activations[i]
            except:
                activation = self.activations[0]

            try:
                kernel_initializer = self.default_kernel_initializer[i]
            except:
                kernel_initializer = self.default_kernel_initializer
                
            try:
                bias_initializer = self.default_bias_initializer[i]            
            except:
                bias_initializer = self.default_bias_initializer

            try:
                activity_regularizer = self.default_activity_regularizer[i]
            except:
                activity_regularizer = self.default_activity_regularizer
                
            layer = layers.Dense(
                neuron, activation=activation,
                kernel_initializer=copy.deepcopy(kernel_initializer),
                bias_initializer=copy.deepcopy(bias_initializer),
                activity_regularizer=copy.deepcopy(activity_regularizer),
                use_bias=self.use_bias)

            if self.dropout_rate > 0.0:
                model.add(layers.Dropout(self.dropout_rate))
            if self.batch_norm:         
                model.add(layers.BatchNormalization(
                    epsilon=self.batch_norm_epsilon))
                
            model.add(layer)
            
        # Add the output layers
        if self.dropout_rate > 0.0:
            model.add(layers.Dropout(self.dropout_rate))        
        if self.batch_norm:
            model.add(layers.BatchNormalization(epsilon=self.batch_norm_epsilon))
            
        output_layer = layers.Dense(self.N_OUTPUT, activation='linear',
                                    kernel_initializer=copy.deepcopy(
                                        self.out_layer_kernel_initializer),
                                    bias_initializer=copy.deepcopy(
                                        self.out_layer_bias_initializer),
                                    use_bias=self.use_bias)
        model.add(output_layer)
      
        model.compile(optimizer=getattr(optimizers, self.optimizer)(),
                      metrics=[getattr(metrics, metric) for metric in self.metrics],
                      loss=getattr(losses, self.loss)(), run_eagerly=True)
        return model
    
