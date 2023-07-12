import java.util.Random;

public class Layer {
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
