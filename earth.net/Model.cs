using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    public class Model
    {


        public Model(double[][] x, double[] y)
        {
            Regressors = x;
            Y = y;
            _rn = Regressors.Length;
            _cn = Regressors[0].Length;
        }

        private int _rn;
        private int _cn;

        private List<Basis> _basises = new List<Basis>();

        public List<Basis> Basises
        {
            get { return _basises; }
            set
            {
                _basises = value;
            }
        }

        public void AddBasis(Basis b)
        {
            _basises.Add(b);
            Recalc();
        }

        public void ResetBasis(IEnumerable<Basis> b)
        {
            _basises.Clear();
            _basises.AddRange(b);
            Recalc();
        }

        public void AddBasis(Basis b, Basis bReflected)
        {
            _basises.Add(b);

            //когда добавляется константа
            if(bReflected != null)
                _basises.Add(bReflected);
            Recalc();
        }

        public void RemoveBasisAt(int basisNumber)
        {
            _basises.RemoveAt(basisNumber);
            Recalc();
        }

        public double[][] Regressors { get; set; }
        public double[] Y { get; set; }
        public double[][] RegressorsTransformed { get; set; }
        
        public double RSS
        {
            get { return _RSS; }
        }

        public double GCV
        {
            get 
            {
                int nBas = this.Basises.Count;
                int nP = Y.Length;

                double effNum = nBas + 3 * (nBas - 1) / 2;
                return _RSS /  (nP * Math.Pow( (1-effNum/nP), 2.0));
            }
        }

        public double RSq
        {
            get
            {
                return _RSq;
            }
        }

        private double _RSS;
        private double _RSq;
        private List<double> _regressionCoefficients;
        //public double[] YTransformed { get; set; }

        public double CheckNewBasis(Basis basis, Basis basisReflected)
        {
                List<Basis> tempNewBasises = new List<Basis>(this.Basises);

                tempNewBasises.Add(basis);
                tempNewBasises.Add(basisReflected);

                var transformedData = Recalc(tempNewBasises, Regressors);
                var tempNewRegressionCoefficients = RegressionToolkit.CalculateLeastSquares(transformedData, Y);
                var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
                var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);
                
                return tempNewRSS;
        }

        public static int _warns = 0;

        public double CheckNewBasisFast(Basis basis, Basis basisReflected, double newKnotVal, Func<double[][], double[], double[]> solver, ref double [][] transformedData)
        {
            if (transformedData == null)
            {
                List<Basis> tempNewBasises = new List<Basis>(this.Basises);
                tempNewBasises.Add(basis);
                tempNewBasises.Add(basisReflected);
                transformedData = Recalc(tempNewBasises, Regressors);
            }
            else
            {

                int ncol = transformedData[0].Length-2;
                int nrow = transformedData.Length;

                bool b1Warning = true;
                bool b2Warning = true;

                double b1Temp = 0.0;
                double b2Temp = 0.0;

                for (int i = 0; i < nrow; i++)
                {
                    transformedData[i][ncol] = basis.Calc(Regressors[i]);
                    transformedData[i][ncol + 1] = basisReflected.Calc(Regressors[i]);
                    if (b1Warning)
                    {
                        b1Warning = b1Temp == transformedData[i][ncol];
                        b1Temp = transformedData[i][ncol];
                    }

                    if (b2Warning)
                    {
                        b2Warning = b2Temp == transformedData[i][ncol];
                        b2Temp = transformedData[i][ncol];
                    }
                    
                }

                if (b1Warning || b2Warning)
                    return double.MaxValue;
                    //Console.WriteLine("Warn {0} {1} !" + ++_warns, b1Temp, b2Temp);
            }

            //var tempNewRegressionCoefficients = RegressionToolkit.CalculateLeastSquares(transformedData, Y);
            var tempNewRegressionCoefficients = solver(transformedData, Y);
            var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
            var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);
            return tempNewRSS;
        }

        public double CheckNewBasisCholeskyFast(Basis basis, Basis basisReflected, double newKnotVal, ref double[][] transformedData)
        {
            return CheckNewBasisFast(basis, basisReflected, newKnotVal, PrepareAndCalcCholessky, ref transformedData);
        }

        public double[] GetRegressorsColumn(int c)
        {
            double [] result = new double[Regressors.Length];
            for (int i = 0; i < Regressors.Length; i++)
                result[i] = Regressors[i][c];
            return result.ToArray();
        }

        public double[] GetRegressorsTransformedColumn(int c)
        {
            double[] result = new double[RegressorsTransformed.Length];
            for (int i = 0; i < RegressorsTransformed.Length; i++)
                result[i] = RegressorsTransformed[i][c];
            return result.ToArray();
        }

        public double[] PrepareAndCalcCholessky(double[][] v, double[] c, double[] means, double yAverage)
        {
            for (int i = 0; i < v.Length; i++)
                v[i][i] += 0.001;

            var regressionCoefficients = RegressionToolkit.CalculateCholesskyRegression(v, c);
            regressionCoefficients[0] = yAverage;

            for (int i = 1; i < regressionCoefficients.Count; i++)
                regressionCoefficients[0] -= regressionCoefficients[i] * means[i];

            return regressionCoefficients.ToArray();
        }

        public double[] PrepareAndCalcCholessky(double[][] x, double[] y)
        {
            double[] bMeans = null;

            var v = __calcV(x, out bMeans);
            var c = __calcC(x);

            return PrepareAndCalcCholessky(v, c, bMeans, y.Average());
        }

        public double CheckNewBasisCholessky(Basis basis, Basis basisReflected)
        {
            List<Basis> tempNewBasises = new List<Basis>(this.Basises);

            tempNewBasises.Add(basis);
            tempNewBasises.Add(basisReflected);

            var transformedData = Recalc(tempNewBasises, Regressors);

            var tempNewRegressionCoefficients = PrepareAndCalcCholessky(transformedData, Y);
            var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
            var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);

            return tempNewRSS;
        }
        

        public void Recalc()
        {
           RegressorsTransformed = Recalc(this.Basises, Regressors);
           //_regressionCoefficients = RegressionToolkit.CalculateLeastSquares(RegressorsTransformed, Y);
           _regressionCoefficients = PrepareAndCalcCholessky(RegressorsTransformed, Y).ToList();
           var predicted = RegressionToolkit.Predict(_regressionCoefficients.ToArray(), RegressorsTransformed);
           _RSS = RegressionToolkit.CalcRSS(predicted.ToArray(), Y);
           _RSq = RegressionToolkit.CalcRSq(predicted.ToArray(), Y);
        }

        public double[] __calcC(double[][] bx)
        {
            var nFeatures = bx[0].Length;
            double[] c = new double[nFeatures];
            double yAvg = Y.Average();

            for (int k = 0; k < bx.Length; k++)
            {

                for (int i = 0; i < nFeatures; i++)
                {
                    c[i] += (Y[k] - yAvg) * bx[k][i];
                }
            }
            return c;
        }

        public double[][] __calcV(double[][] bx, out double [] means)
        {
            var nFeatures = bx[0].Length;
            means = new double[nFeatures];
            double[][] v = new double[nFeatures][];
            for (int i = 0; i < nFeatures; i++)
                v[i] = new double[nFeatures];

            for (int k = 0; k < bx.Length; k++)
            {
                for (int i = 0; i < nFeatures; i++)
                {
                    for (int j = 0; j < bx.Length; j++)
                        means[i] += bx[j][i];
                    means[i] = means[i] / bx.Length;

                    for (int j = 0; j < nFeatures; j++)
                    {
                        v[i][j] += bx[k][j] * (bx[k][i] - means[i]);
                    }
                }
            }
            return v;
        }


        private class comparerArrayOrder : IComparer<KeyValuePair<double, int>>
        {
            public int Compare(KeyValuePair<double, int> x, KeyValuePair<double, int> y)
            {
                int c = x.Key.CompareTo(y.Key);

                if (c == 0)
                    return (x.Value.CompareTo(y.Value));
                else return c;
            }
        }

        public int[] GetArrayOrder(double[] x)
        {
            var uns = x.ToList();
            SortedDictionary<KeyValuePair<double, int>, int> t = new SortedDictionary<KeyValuePair<double, int>, int>(new comparerArrayOrder());
            for (int i = 0; i < x.Length; i++)
                t.Add(new KeyValuePair<double, int>(x[i], i), i);
            return t.Values.ToArray();
        }

        public double[][] Recalc(List<Basis> basises, double[][] regressors)
        {
            double [][] resultDataset;
            resultDataset = new double[_rn][];
            //YTransformed = new double[_rn];
            for (int i = 0; i < resultDataset.Length; i++)
            {
                resultDataset[i] = new double[basises.Count];
                for (int j = 0; j < basises.Count; j++)
                {
                    resultDataset[i][j] = basises[j].Calc(regressors[i]);
                    //YTransformed[i] += RegressorsTransformed[i][j];
                }
            }
            return resultDataset;
        }

        internal double __CalcCFast(Basis b, Basis bReflected , int k, int iParent, int iFeature ,int mNew, int [] kOrdered, ref double[] c, ref double ybxSum)
        {
            List<Basis> tempNewBasises = new List<Basis>(this.Basises);
            tempNewBasises.Add(b);
            tempNewBasises.Add(bReflected);
            var transformedData = Recalc(tempNewBasises, Regressors);

            var ix0 = kOrdered[k];     //x0
            var ix1 = kOrdered[k + 1]; //x1

            //int mNew = transformedData[0].Length - 1;
            
            var x0 = Regressors[ix0][iFeature];
            var x1 = Regressors[ix1][iFeature];
            var bx1 = transformedData[ix1][iParent];
            var xDelta = x1 - x0;

            if (c == null)
                c = __calcC(transformedData);
            else
            {
                ybxSum += (Y[ix1] - Y.Average())*bx1;
                c[mNew] += xDelta * ybxSum;
            }

            //THE TESTING SHIT
            var chat = __calcC(transformedData);
            for (int i = 0; i < c.Length; i++ )
                Console.WriteLine("{0, 20} {1, 20}", chat[i], c[i]);
            return 0.0;
        }
    }
}
