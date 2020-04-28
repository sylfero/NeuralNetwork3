using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork3.InputFunctions
{
    class WeightedSumFunction : IInputFunction
    {
        public double Calculate(List<Synapse> inputs) => inputs.Select(x => x.Weight * x.Output).Sum();
    }
}
