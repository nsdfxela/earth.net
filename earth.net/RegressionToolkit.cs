﻿using MathNet.Numerics;
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
        public static double CalcRSS(double[] yhat, double[] y)
        {
            double result = 0.0;

            for (int i = 0; i < y.Length; i++)
            {
                double diff = Math.Pow(y[i] - yhat[i], 2);
                result += diff;
            }
            return result;
        }

        public static double CalcRSq(double[] yhat, double[] y)
        {
            var rss = CalcRSS(yhat, y);
            var yAvg = y.Average();
            return 1 - rss / (y.Select(v => Math.Pow(v - yAvg, 2)).Sum());
        }

        public static List<double> Predict(double[] caffs, double[][] xValues)
        {
            Matrix<double> x = Matrix<double>.Build.DenseOfRowArrays(xValues);
            Matrix<double> k = Matrix<double>.Build.DenseOfColumnArrays(caffs);
            var resultMatrix = x * k;
            List<double> result = resultMatrix.Column(0).ToList();
            return result;
        }
        public static int _good = 0;
        public static int _bad = 0;

        public static List<double> CalculateCholesskyRegression(double [][] v, double [] c)
        {
            double[] slopes = null;

            try
            {
                slopes = MultipleRegression.NormalEquations(v, c, false);
            }
            catch
            {
                slopes = MultipleRegression.QR(v, c);
                Console.WriteLine("Unable to solve with Cholessky" + ++_bad);
            }
            if (slopes.Any(s => double.IsNaN(s)))
            {
                for (int i = 0; i < slopes.Length; i++)
                    if (double.IsInfinity(slopes[i]) || double.IsNaN(slopes[i]))
                        slopes[i] = 0.0;
            }
            return slopes.ToList();

        }

        public static List<double> CalculateLeastSquares(double[][] x, double[] y)
        {
            double[] slopes = null;
            try
            {
                slopes = MultipleRegression.QR(x, y, false);

                if (slopes.Any(s => double.IsNaN(s)))
                {
                    for (int i = 0; i < slopes.Length; i++)
                        if (double.IsInfinity(slopes[i]) || double.IsNaN(slopes[i]))
                            slopes[i] = 0.0;
                }
                return slopes.ToList();
            }
            catch
            {
                Console.WriteLine(DoubleToR(x));
                Console.WriteLine(DoubleToR(y));
                Console.WriteLine(slopes);
                return null;
            }
        }

        public static string DoubleToR(double [][]x)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("mtrx <- matrix(c(");

            for (int i = 0; i < x.Length; i++)
                for (int j = 0; j < x[i].Length; j++)
                {
                    sb.Append(String.Format(" {0},", x[i][j].ToString().Replace(',','.') ));
                }

            sb = new StringBuilder( sb.ToString().TrimEnd(new char[] {','}));

            sb.Append(string.Format("), nrow ={0}, byrow=TRUE)", x.Length));
            
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
