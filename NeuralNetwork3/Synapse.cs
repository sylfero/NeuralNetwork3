using System;

namespace NeuralNetwork3
{
    class Synapse
    {
        private Neuron fromNeuron;
        private Neuron toNeuron;

        public double Weight { get; set; }
        public double Output { get; set; }

        private static readonly Random rnd = new Random();

        public Synapse(Neuron fromNeuron, Neuron toNeuron)
        {
            this.fromNeuron = fromNeuron;
            this.toNeuron = toNeuron;
            Weight = rnd.NextDouble();
        }

        public void UpdateWeight(double learnRate, double delta)
        {
            Weight += delta;
        }
    }
}
