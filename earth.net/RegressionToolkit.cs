using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearRegression;
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
            try
            {
                
                var slopes = MultipleRegression.QR(x, y, true);
                return slopes.ToList();
            }
            catch
            {
                Console.WriteLine(DoubleToR(x));
                Console.WriteLine(DoubleToR(y));
                return null;
            }
        }

        public static string DoubleToR(double [][]x)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("mtrx <- as.matrix(c(");

            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < x[i].Length; j++)
                {
                    sb.Append(String.Format(" {0},", x[i][j].ToString().Replace(',','.') ));
                }

            sb = new StringBuilder( sb.ToString().TrimEnd(new char[] {','}));
            sb.Append("))");
            sb.AppendLine(string.Format("dim(mtrx) <- c({0},{1})", x.Length, x[0].Length));
            
            return sb.ToString();
        }

        public static string DoubleToR(double[] y)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Y <- as.matrix(c(");

            for (int i = 0; i < y.Length; i++)
                {
                    sb.Append(String.Format("{0},", y[i].ToString().Replace(',', '.')));
                }

            sb = new StringBuilder(sb.ToString().TrimEnd(new char[] { ',' }));
            sb.Append("))");
            return sb.ToString();
        }


    }
}
