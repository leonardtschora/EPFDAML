from work.models.NeuralNetworks.NeuralNetworks import *

class LeNet(NeuralNetwork):
    def __init__(self, name, model_, W, H):
        NeuralNetwork.__init__(self, name, model_)
        
        self.conv_activation = model_["conv_activation"]
        self.filter_size = model_["filter_size"]
        self.dilation_rate = model_["dilation_rate"]
        self.kernel_size = model_["kernel_size"]
        self.pool_size = model_["pool_size"]
        self.strides = model_["strides"]

        self.W = W
        self.H = H

    def create_network(self, input_shape=None):        
        model = tf.keras.Sequential([Input(shape=input_shape)])
        
        conv_activation = self.conv_activation
        for i in range(len(self.filter_size)):
            for j in range(len(self.filter_size[i])):
                filter_size = self.filter_size[i][j]
                dilation_rate = self.dilation_rate[i][j]
                kernel_size = self.kernel_size[i][j]
                model.add(Conv2D(filter_size, kernel_size=kernel_size, padding="same",
                                 dilation_rate=dilation_rate,
                                 activation=conv_activation))
                
            pool_size = self.pool_size[i]
            strides = self.strides[i]
            if pool_size is not None:
                model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
        
        model.add(Flatten())

        # Add the output dense layers
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
                
            layer = layers.Dense(neuron, activation=activation,
                                 kernel_initializer=copy.deepcopy(kernel_initializer),
                                 bias_initializer=copy.deepcopy(bias_initializer),
                                 activity_regularizer=copy.deepcopy(activity_regularizer),
                                 use_bias=self.use_bias)

            if self.dropout_rate > 0.0:
                model.add(layers.Dropout(self.dropout_rate))
            if self.batch_norm:         
                model.add(layers.BatchNormalization(epsilon=self.batch_norm_epsilon))
                
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
        
    def fit(self, X, y, verbose=0, Xv=None, yv=None):
        # Reshape the input data
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        
        X = X.reshape(-1, self.W, self.H)
        X = np.expand_dims(X, -1)

        Xv = Xv.reshape(-1, self.W, self.H)
        Xv = np.expand_dims(Xv, -1)        
        
        self.update_params(input_shape=X.shape[1:])        
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, validation_data=(Xv, yv),
                       shuffle=self.shuffle_train, verbose=verbose)
        return self

    def predict(self, X):
        X = X.reshape(-1, self.W, self.H)
        X = np.expand_dims(X, -1)
        
        return tf.reshape(self.model.predict_step(X), (-1, self.N_OUTPUT))
