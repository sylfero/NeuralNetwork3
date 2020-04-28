using System;

namespace NeuralNetwork3.ActivationFunctions
{
    class SigmoidActivationFunction : IActivationFunction
    {
        public double Calculate(double input) => 1 / (1 + Math.Exp(-input));

        public double Derivative(double input) => Calculate(input) * (1 - Calculate(input));
    }
}
