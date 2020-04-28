using System;
using System.IO;

namespace NeuralNetwork3
{
    static class Data
    {
        public static double[][] Get(string path)
        {
            string[] lines = File.ReadAllLines(path);
            double[][] data = new double[lines.Length][];

            for (int i = 0; i < lines.Length; i++)
            {
                string[] tmp = lines[i].Split(',');
                data[i] = new double[tmp.Length + 2];

                for (int j = 0; j < tmp.Length - 1; j++)
                {
                    data[i][j] = Double.Parse(tmp[j]);
                }
                int last = tmp.Length - 1;
                if (tmp[last].Equals("Iris-setosa"))
                {
                    data[i][last] = 1;
                    data[i][last + 1] = 0;
                    data[i][last + 2] = 0;
                }
                else if (tmp[last].Equals("Iris-versicolor"))
                {
                    data[i][last] = 0;
                    data[i][last + 1] = 1;
                    data[i][last + 2] = 0;
                }
                else
                {
                    data[i][last] = 0;
                    data[i][last + 1] = 0;
                    data[i][last + 2] = 1;
                }
            }

            return data;
        }

        public static void Shuffle(this double[][] data)
        {
            Random rnd = new Random();
            for (int i = 0; i < data.Length; i++)
            {
                double[] tmp = data[i];
                int r = rnd.Next(i, data.Length);
                data[i] = data[r];
                data[r] = tmp;
            }
        }

        public static void Normalize(this double[][] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                double max = data[i][0];
                double min = data[i][0];
                for (int j = 1; j < data[i].Length - 3; j++)
                {
                    if (data[i][j] > max)
                    {
                        max = data[i][j];
                    }
                    else if (data[i][j] < min)
                    {
                        min = data[i][j];
                    }
                }
                for (int j = 0; j < data[i].Length - 3; j++)
                {
                    data[i][j] = (data[i][j] - min) / (max - min);
                }
            }
        }

        public static void Normalize(this double[] data)
        {
            double max = data[0];
            double min = data[0];
            for (int i = 1; i < data.Length; i++)
            {
                if (data[i] > max)
                {
                    max = data[i];
                }
                else if (data[i] < min)
                {
                    min = data[i];
                }
            }
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = (data[i] - min) / (max - min);
            }
        }

        public static double[][] GetInputs(this double[][] data)
        {
            double[][] outArray = (double[][])data.Clone();
            for (int i = 0; i < data.Length; i++)
            {
                Array.Resize(ref outArray[i], outArray[i].Length - 3);
            }

            return outArray;
        }

        public static double[][] GetOutputs(this double[][] data)
        {
            double[][] outArray = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                outArray[i] = new double[3];
                for (int j = data[i].Length - 3; j < data[i].Length; j++)
                {
                    outArray[i][j - 4] = data[i][j];
                }
            }

            return outArray;
        }
    }
}
