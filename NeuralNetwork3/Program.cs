using NeuralNetwork3.ActivationFunctions;
using NeuralNetwork3.InputFunctions;
using System;

namespace NeuralNetwork3
{
    class Program
    {
        static void Main(string[] args)
        {
            double[][] data = Data.Get("data.txt");
            data.Normalize();
            data.Shuffle();
            double[][] trainData = data.GetInputs();
            double[][] expectedValues = data.GetOutputs();

            Network network = new Network(1, new WeightedSumFunction(), new SigmoidActivationFunction(), 4, 3, 3);
            network.Train(trainData, expectedValues, 1000);

            double[] d = new double[] { 5.1, 3.5, 1.4, 0.2 };
            d.Normalize();
            double[] final = network.Calculate(d);
            foreach (var i in final)
            {
                Console.WriteLine(i);
            }
            Console.WriteLine();

            d = new double[] { 5.8, 2.8, 5.1, 2.4 };
            d.Normalize();
            final = network.Calculate(d);
            foreach (var i in final)
            {
                Console.WriteLine(i);
            }
            Console.WriteLine();

            d = new double[] { 6.1, 2.8, 4.7, 1.2 };
            d.Normalize();
            final = network.Calculate(d);
            foreach (var i in final)
            {
                Console.WriteLine(i);
            }
            Console.ReadKey();
        }
    }
}
