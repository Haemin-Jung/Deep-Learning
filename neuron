class neuron():
    
    def __init__(self):
        self.w = 1.0
        self.b = 1.0
        
    def feed_forward(self, x):
        y_hat = x * self.w + self.b
        return y_hat
    
    def back_propagate(self, x_i, y_i, y_hat):
        error = y_i - y_hat
        self.w = self.w + error*x_i
        self.b = self.b + error*1

    def fit(self, x, y, epoch):
        for i in range(0, epoch):
            for x_i, y_i in zip(x, y):
                y_hat = self.feed_forward(x_i)
                self.back_propagate(x_i, y_i, y_hat)
