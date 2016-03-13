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

        public double CheckNewBasisFast(Basis basis, Basis basisReflected, double newKnotVal, ref double [][] transformedData)
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

                int ncol = transformedData[0].Length;
                int nrow = transformedData.Length;

                for (int i = 0; i < nrow; i++)
                {
                    double[] newdata = new double[ncol + 2];

                    for (int j = 0; j < ncol; j++)
                    {
                        newdata[j] = transformedData[i][j];
                    }
                    newdata[ncol] = basis.Calc(Regressors[i]);
                    newdata[ncol + 1] = basisReflected.Calc(Regressors[i]);
                }
            }

            var tempNewRegressionCoefficients = RegressionToolkit.CalculateLeastSquares(transformedData, Y);
            var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
            var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);
            return tempNewRSS;
        }

        public double CheckNewBasisCholessky(Basis basis, Basis basisReflected)
        {
            List<Basis> tempNewBasises = new List<Basis>(this.Basises);

            tempNewBasises.Add(basis);
            tempNewBasises.Add(basisReflected);

            var transformedData = Recalc(tempNewBasises, Regressors);
            
            double [] bMeans;

            var v = __calcV(transformedData, out bMeans);
            var c = __calcC(transformedData);

            for (int i = 0; i < v.Length; i++)
                v[i][i] += 0.001;
            
            var tempNewRegressionCoefficients = RegressionToolkit.CalculateCholesskyRegression(v, c);
            tempNewRegressionCoefficients[0] = Y.Average();
            
            for (int i = 1; i < tempNewRegressionCoefficients.Count; i++)
                tempNewRegressionCoefficients[0] -= tempNewRegressionCoefficients[i] * bMeans[i];

            var tempNewpredicted = RegressionToolkit.Predict(tempNewRegressionCoefficients.ToArray(), transformedData);
            var tempNewRSS = RegressionToolkit.CalcRSS(tempNewpredicted.ToArray(), Y);

            return tempNewRSS;
        }

        public void Recalc()
        {
           RegressorsTransformed = Recalc(this.Basises, Regressors);
           _regressionCoefficients = RegressionToolkit.CalculateLeastSquares(RegressorsTransformed, Y);
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
    }
}
