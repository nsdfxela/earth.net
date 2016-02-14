using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    class RegressionToolkit
    {
        public static double calcRSS(double[] yhat, double[] y)
        {
            double result = 0.0;

            for (int i = 0; i < y.Length; i++)
            {
                double diff = Math.Pow(y[i] - yhat[i], 2);
                result += diff;
            }
            return result;
        }


        public static List<double> Predict(double[] caffs, double[][] xValues)
        {
            Matrix<double> x = Matrix<double>.Build.DenseOfRowArrays(xValues);
            Matrix<double> k = Matrix<double>.Build.DenseOfColumnArrays(caffs);
            var b = k.Row(0);
            k = k.RemoveRow(0);
            var resultMatrix = x * k;
            List<double> result = resultMatrix.Column(0).ToList();
            for (int i = 0; i < result.Count; i++)
                result[i] = result[i] + b[0];

            return result;
        }

        public static List<double> CalculateLeastSquares(double[][] x, double[] y)
        {
            var slopes = Fit.MultiDim(x, y, true);
            return slopes.ToList();
        }

    }
}
