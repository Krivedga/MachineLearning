import java.util.Random;

public class Main {

    public static void main(String[] args) {
        Random random = new Random(0);
        Network network = new Network(0.1, 1, 1,2, 2);
        double[] inputs = null;
        for (int i = 0; i < 2500; i++) {
            inputs = new double[]{random.nextDouble()};
            network.giveInput(inputs);
            network.backPropagation(inputs[0] > 0.5 ? new double[]{1, 0} : new double[]{0, 1});
        }
        for (double i = 0; i < 1d; i+= 0.03d) {
            inputs = new double[]{i};
            double[] outputs = network.giveInput(inputs);
            if (outputs[0] > outputs[1]) {
                System.out.println((int)(inputs[0]*1000)+" > 500");
            } else {
                System.out.println((int)(inputs[0]*1000)+" <= 500");
            }
        }
//        network.debug();

    }


}
