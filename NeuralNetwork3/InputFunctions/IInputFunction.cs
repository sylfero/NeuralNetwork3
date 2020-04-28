using System.Collections.Generic;

namespace NeuralNetwork3.InputFunctions
{
    interface IInputFunction
    {
        double Calculate(List<Synapse> inputs);
    }
}
