import java.util.Random;

public class Network {
    private class Layer {
        private static Random random = new Random(0);
        public double[] neurons;
        public double[][] weights;
        public double bias[];
        public int size;

        public Layer(int size, int nextSize) {
            this.size = size;
            neurons = new double[size];
            weights = new double[size][nextSize];
            bias = new double[size];
            for (int i = 0; i < size; i++) {
                bias[i] = random.nextDouble() * 2 - 1;
                neurons[i] = 0;
                for (int j = 0; j < nextSize; j++) {
                    weights[i][j] = random.nextDouble() - 0.5d;
                }
            }
        }
    }

    private Layer[] layers;
    private final double learningRate;
    private final double moment;

    public Network(double learningRate, double moment, int... sizes) {
        this.learningRate = learningRate;
        this.moment = moment;
        layers = new Layer[sizes.length];
        for (int i = 0; i < sizes.length; i++) {
            if (i == sizes.length - 1) {
                layers[i] = new Layer(sizes[i], 0);
                break;
            }
            layers[i] = new Layer(sizes[i], sizes[i + 1]);
        }
    }

    public double[] giveInput(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int l = 1; l < layers.length; l++) {
            Layer rightLayer = layers[l];
            Layer leftLayer = layers[l - 1];
            for (int j = 0; j < rightLayer.size; j++) {
                double s = 0;
                for (int k = 0; k < leftLayer.size; k++) {
                    s += leftLayer.weights[k][j] * leftLayer.neurons[k];
                }
                s += rightLayer.bias[j];
                s = sigmoid(s);
                rightLayer.neurons[j] = s;
            }
        }
        return layers[layers.length - 1].neurons;
    }

    public void backPropagation(double[] targets) {
        double[] errors;
        Layer lastLayer = layers[layers.length - 1];
        errors = new double[lastLayer.neurons.length];

        for (int i = 0; i < lastLayer.neurons.length; i++) {
            errors[i] = error(lastLayer.neurons[i], targets[i]);
        }
        for (int l = layers.length - 2; l >= 0; l--) {
            Layer leftLayer = layers[l];
            Layer rightLayer = layers[l + 1];
            double[] nextErrors = new double[leftLayer.size];
            for (int ln = 0; ln < leftLayer.size; ln++) {
                double s = 0;
                for (int rn = 0; rn < rightLayer.size; rn++) {
                    s += leftLayer.weights[ln][rn] * errors[rn];
                }
                s *= sigmoidDerivative(leftLayer.neurons[ln]);
                nextErrors[ln] = s;
            }
            for (int ln = 0; ln < leftLayer.size; ln++) {
                for (int rn = 0; rn < rightLayer.size; rn++) {
                    leftLayer.weights[ln][rn] = leftLayer.weights[ln][rn] - leftLayer.neurons[ln] * errors[rn] * learningRate;
                }
            }
            for (int rn = 0; rn < rightLayer.size; rn++) {
                rightLayer.bias[rn] = rightLayer.bias[rn] - errors[rn] * learningRate;
            }
            errors = new double[nextErrors.length];
            System.arraycopy(nextErrors, 0, errors, 0, nextErrors.length);
        }
    }

    public double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double y) {
        return y * (1 - y);
    }

    public double error(double got, double expected) {
        return got - expected;
    }

}
