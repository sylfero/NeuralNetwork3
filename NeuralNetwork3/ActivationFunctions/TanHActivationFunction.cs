using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork3.ActivationFunctions
{
    class TanHActivationFunction : IActivationFunction
    {
        public double Calculate(double input) => (2 / (1 + Math.Exp(-2 * input))) - 1;

        public double Derivative(double input) => 1 - Math.Pow(Calculate(input), 2);
    }
}
