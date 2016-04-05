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
        
        public double uValue1 = double.MinValue;
        public double uValue2 = double.MinValue;

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
                    transformedData[i][ncol] += basis.CalcFastDependedOnPrevious(Regressors[i], newKnotVal, basis.Hinges[basis.Hinges.Count-1].Value, i);
                    transformedData[i][ncol+1] += basisReflected.CalcFastDependedOnPrevious(Regressors[i], newKnotVal, basis.Hinges[basis.Hinges.Count - 1].Value, i);

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
            //return CheckNewBasisFast(basis, basisReflected, newKnotVal, PrepareAndCalcCholesskyFull, ref transformedData);
            return CheckNewBasisFast(basis, basisReflected, newKnotVal, PrepareAndCalcCholesskyNewColumns, ref transformedData);
        }

        public double[] GetRegressorsColumn(int c)
        {
            double [] result = new double[Regressors.Length];
            for (int i = 0; i < Regressors.Length; i++)
                result[i] = Regressors[i][c];
            return result.ToArray();
        }

        /// <summary>
        /// temp V
        /// </summary>
        double[][] _v;
        /// <summary>
        /// temp C
        /// </summary>
        double[] _c;

        List<Double> xhat = new List<double>(new double[] { 0.0 });
        

        double __calcMean(double [][] x, int f)
        {
            double res = 0.0;
            double len = x.Length;

            for (int i = 0; i < x.Length; i++)
                res += x[i][f];

            
            return res/len;
        }
        
        /// <summary>
        /// Resize V-mtrx
        /// </summary>
        /// <param name="v">initial matrix</param>
        /// <param name="addCount">number of rows and colls to add</param>
        /// <returns>resized V</returns>
        public static double[][] ResizeV(double[][] v, int newSize)
        {
            if (v == null)
            {
                double[][] nv = new double[newSize][];
                for (int i = 0; i < newSize; i++)
                    nv[i] = new double[newSize];
                return nv;
            }

            //int newSize = v.Length + addCount;
            double[][] newV = new double[newSize][];
            for (int i = 0; i < newV.Length; i++)
            {
                if (i < v.Length)
                {
                    newV[i] = v[i];
                    Array.Resize(ref  newV[i], newSize);
                }
                else
                    newV[i] = new double[newSize];
            }
            return newV;
        }

        public double[] PrepareAndCalcCholesskyNewColumns(double[][] x, double[] y)
        {
            //Это должно вызываться после пересчета базисов с новым узлом
            if (_v.Length != x[0].Length ||
                _v[0].Length != x[0].Length ||
                    _c.Length != x[0].Length || 
                        _v.Length < 3)
            { 
                //TODO: Не надо пересчитывать полностью, а только добавленные на предыдущей операции колонки 
                _v = __calcV(x, out xhat);
            }
            else
            {
                //ПЕРЕСЧИТАТЬ ТОЛЬКО ПОСЛЕДНИЕ ДВА СТОЛБЦА И ПОСЛЕДНИЕ ДВЕ КОЛОНКИ V
                int f0 = _v.Length - 2;
                int f1 = _v.Length - 1;

                xhat[f0] = __calcMean(x, f0);
                xhat[f1] = __calcMean(x, f1);
                //Зануляем то, что будем считать
                for (int i = f0; i <= f1; i++) //две колонки
                {
                    for (int j = 0; j < _v.Length; j++)
                    {
                        _v[i][j] = 0.0;
                        _v[j][i] = 0.0;
                    }
                }

                //считаем
                for (int k = 0; k < x.Length; k++)
                {

                    for (int i = f0; i <= f1; i++) //две колонки
                    {
                        for (int j = 0; j <= i; j++)// все фичи (Bj)
                        {
                            _v[i][j] += x[k][j] * (x[k][i] - xhat[i]);
                            if (i != j)
                                _v[j][i] += x[k][i] * (x[k][j] - xhat[j]);
                        }
                    }
                }

            }
            //ВЗЯТО ИЗ СТАРОЙ РЕАЛИЗАЦИИ (пока)
            for (int i = 0; i < _v.Length; i++)
                _v[i][i] += 0.001;

            //СОПОСТАВЛЕНИЕ
            //for (int i = 0; i < _v.Length; i++)
            //    for (int j = 0; j < _v.Length; j++)
            //    {
            //        var d1 = _v[i][j] - VDEBUG[i][j];
            //        var d2 = _v[j][i] - VDEBUG[j][i];
            //        if (Math.Abs(d1) >= 0.1 || Math.Abs(d2) >= 0.1)
            //        {
            //            Console.WriteLine("DEBUG : {0}", d1);
            //            Console.WriteLine("DEBUG : {0}", d2);
            //        }
            //    }

            _c = __calcC(x);

            var regressionCoefficients = RegressionToolkit.CalculateCholesskyRegression(_v, _c);
            regressionCoefficients[0] = y.Average();

            for (int i = 1; i < regressionCoefficients.Count; i++)
                regressionCoefficients[0] -= regressionCoefficients[i] * xhat[i];

            return regressionCoefficients.ToArray();
        }

        public double[] PrepareAndCalcCholesskyFull(double[][] x, double[] y)
        {
            List<double> bMeans;

            _v = __calcV(x, out bMeans);
            _c = __calcC(x);

            for (int i = 0; i < _v.Length; i++)
                _v[i][i] += 0.001;

            var regressionCoefficients = RegressionToolkit.CalculateCholesskyRegression(_v, _c);
            regressionCoefficients[0] = y.Average();

            for (int i = 1; i < regressionCoefficients.Count; i++)
                regressionCoefficients[0] -= regressionCoefficients[i] * bMeans[i];

            return regressionCoefficients.ToArray();
        }

        public double CheckNewBasisCholessky(Basis basis, Basis basisReflected)
        {
            List<Basis> tempNewBasises = new List<Basis>(this.Basises);

            tempNewBasises.Add(basis);
            tempNewBasises.Add(basisReflected);

            var transformedData = Recalc(tempNewBasises, Regressors);

            var tempNewRegressionCoefficients = PrepareAndCalcCholesskyFull(transformedData, Y);
            var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
            var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);

            return tempNewRSS;
        }
        
        public void Recalc()
        {
           RegressorsTransformed = Recalc(this.Basises, Regressors);
           //_regressionCoefficients = RegressionToolkit.CalculateLeastSquares(RegressorsTransformed, Y);
           //regressionCoefficients = PrepareAndCalcCholesskyFull(RegressorsTransformed, Y).ToList();
           
            //Задать V нужный размер
           _v = ResizeV(this._v, RegressorsTransformed[0].Length);
            //А C можно пересчитать прямо тут
           _c = __calcC(RegressorsTransformed);

           _regressionCoefficients = PrepareAndCalcCholesskyNewColumns(RegressorsTransformed, Y).ToList();
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

        public double[][] __calcV(double[][] bx, out List<Double> means)
        {
            var nFeatures = bx[0].Length;
            means = new List<double>(nFeatures);
            
            for (int n = 0; n < nFeatures; n++)
                means.Add(__calcMean(bx, n));


            double[][] v = new double[nFeatures][];
            for (int i = 0; i < nFeatures; i++)
                v[i] = new double[nFeatures];

            

            for (int k = 0; k < bx.Length; k++)
            {
                for (int i = 0; i < nFeatures; i++)
                {
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
                    resultDataset[i][j] = basises[j].CalcFast(regressors[i], i);
                    //YTransformed[i] += RegressorsTransformed[i][j];
                }
            }
            return resultDataset;
        }

    }
}
