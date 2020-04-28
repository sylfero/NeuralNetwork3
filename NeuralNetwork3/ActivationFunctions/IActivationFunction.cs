namespace NeuralNetwork3.ActivationFunctions
{
    interface IActivationFunction
    {
        double Calculate(double input);

        double Derivative(double input);
    }
}
