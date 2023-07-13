import java.util.Random;

public class Main {

    public static void main(String[] args) {
        Random random = new Random(0);
        Network network = new Network(0.035, 1, 1,4, 2);
        double[] inputs = null;
        for (int i = 0; i < 90000; i++) {
            inputs = new double[]{random.nextDouble()};
            network.giveInput(inputs);
            network.backPropagation((inputs[0] < 0.8 && inputs[0] > 0.4) ? new double[]{1, 0} : new double[]{0, 1});
        }
        for (double i = 0; i < 1d; i += 0.06d) {
            inputs = new double[]{i};
            double[] outputs = network.giveInput(inputs);
            System.out.println("Condition for " + (int)(inputs[0]*1000) + " is " + (outputs[0] > outputs[1]));
//            if (outputs[0] > outputs[1]) {
//                System.out.println((int)(inputs[0]*1000)+" > 800");
//            } else {
//                System.out.println((int)(inputs[0]*1000)+" <= 800");
//            }
        }
//        network.debug();

    }


}
