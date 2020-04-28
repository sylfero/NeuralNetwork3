using NeuralNetwork3.ActivationFunctions;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork3.InputFunctions;

namespace NeuralNetwork3
{
    class Network
    {
        public List<Layer> Layers { get; set; } = new List<Layer>();

        public double LearningRate { get; set; }

        public Network(double learningRate, IInputFunction inputFunction,IActivationFunction activationFunction, params int[] numberOfNeurons)
        {
            LearningRate = learningRate;

            //numberOfNeurons[] - how many layers there should be and how many neurons each layer should have
            foreach (int number in numberOfNeurons)
            {
                Layer layer = new Layer();
                Layers.Add(layer);

                for (int i = 0; i < number; i++)
                {
                    layer.Neurons.Add(new Neuron(activationFunction, inputFunction));
                }
            }

            ConnectLayers();
        }

        public void Train(double[][] inputValues, double[][] expectedValues, int epochs)
        {
            //We will update only weights of synapses between neurons so errors[0] will be left blank
            double[][] errors = new double[Layers.Count][];
            //Each error[i] has size based on how many Neurons Layer[i] has
            for (int i = 0; i < Layers.Count; i++)
            {
                errors[i] = new double[Layers[i].Neurons.Count];
            }
 
            for (int i = 0; i < epochs; i++)
            {     
                for (int j = 0; j < inputValues.Length; j++)
                {
                    double[] outputs = Calculate(inputValues[j]);

                    CalculateErrors(outputs, expectedValues[j], errors);

                    UpdateWeights(errors);
                }
            }
        }

        public double[] Calculate(double[] input)
        {
            PushInputValues(input);

            //We gather calculated values from last layer neurons to array
            double[] outputs = new double[Layers.Last().Neurons.Count];
            for (int i = 0; i < outputs.Length; i++)
            {
                Neuron currentNeuron = Layers.Last().Neurons[i];
                outputs[i] = currentNeuron.OutputValue;
            }
            return outputs;
        }

        private void PushInputValues(double[] inputs)
        {
            //Give inputs values to first layer of neurons
            for (int i = 0; i < inputs.Length; i++)
            {
                Neuron currentNeuron = Layers.First().Neurons[i];
                currentNeuron.OutputValue = currentNeuron.InputValue = inputs[i];
            }

            //Calculate outputs for other layers
            for (int i = 0; i < Layers.Count; i++)
            {
                //Calculate output for each neuron and give this value to output synapses connected to this neuron
                foreach (Neuron neuron in Layers[i].Neurons)
                { 
                    neuron.PushValueOnOuput(neuron.CalculateOutput());
                }
            }
        }

        private void ConnectLayers()
        {
            //We don't connect first layer to anythin befor so we can omit it
            for (int i = 1; i < Layers.Count; i++)
            {
                //Each neuron in given layer
                foreach (Neuron neuron in Layers[i].Neurons)
                {
                    //Is connected with each nuron in previous layer with synapse
                    foreach (Neuron previousNeuron in Layers[i - 1].Neurons)
                    {
                        Synapse synapse = new Synapse(previousNeuron, neuron);
                        neuron.Inputs.Add(synapse);
                        previousNeuron.Outputs.Add(synapse);
                    }
                }
            }
        }

        private void CalculateErrors(double[] outputs, double[] expectedValues, double[][] errors)
        {
            //Compare outputs with expected values (global error)
            for (int i = 0; i < outputs.Length; i++)
            {
                Neuron currentNeuron = Layers.Last().Neurons[i];
                errors.Last()[i] = currentNeuron.ActivationFunction.Derivative(currentNeuron.InputValue) * (outputs[i] - expectedValues[i]);
            }

            //Based on output layer errors calculate other layers errors (we don't calculate error for first and last layer here)
            for (int i = Layers.Count - 2; i > 0; i--)
            {
                for (int j = 0; j < Layers[i].Neurons.Count; j++)
                {
                    errors[i][j] = 0;
                    //Sum of product of errors from next layer and weights of synapses connecting this two neurons 
                    for (int k = 0; k < Layers[i + 1].Neurons.Count; k++)
                    {
                        errors[i][j] += errors[i + 1][k] * Layers[i + 1].Neurons[k].Inputs[j].Weight;
                    }
                    Neuron currentNeuron = Layers[i].Neurons[j];
                    errors[i][j] *= currentNeuron.ActivationFunction.Derivative(currentNeuron.InputValue);
                }
            }
        }

        private void UpdateWeights(double[][] errors)
        {
            //Update wights for all synapses based on calculated errors
            for (int i = Layers.Count - 1; i > 0; i--)
            {
                for (int j = 0; j < Layers[i].Neurons.Count; j++)
                {
                    for (int k = 0; k < Layers[i - 1].Neurons.Count; k++)
                    {
                        double delta = errors[i][j] * Layers[i - 1].Neurons[k].OutputValue * -1;
                        Layers[i].Neurons[j].Inputs[k].UpdateWeight(LearningRate, delta);
                    }
                }
            }
        }
    }
}
