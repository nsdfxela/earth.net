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

       public const int MAX_TERMS_ALLOWED = 100;

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

        public double[][] RecalcBx(Basis basis, Basis basisReflected, double newKnotVal, ref double[][] transformedData)
        {
            List<Basis> tempNewBasises = new List<Basis>(this.Basises);
            basis.Hinges.AddRange(basisReflected.Hinges);
            tempNewBasises.Add(basis);
            //tempNewBasises.Add(basisReflected);
            transformedData = Recalc(tempNewBasises, Regressors);
            return transformedData;
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

        public void Recalc()
        {
           RegressorsTransformed = Recalc(this.Basises, Regressors);
           _regressionCoefficients = RegressionToolkit.CalculateLeastSquares(RegressorsTransformed, Y);
           var predicted = RegressionToolkit.Predict(_regressionCoefficients.ToArray(), RegressorsTransformed);
           _RSS = RegressionToolkit.CalcRSS(predicted.ToArray(), Y);
           _RSq = RegressionToolkit.CalcRSq(predicted.ToArray(), Y);
        }


        public double[] __getColumn(double[][] matrix, int c)
        {
            //needfaster begin
            List<double> col = new List<double>();
            for (int i = 0; i < matrix.Length;i++ )
                col.Add(matrix[i][c]);
            return col.ToArray();
            //needfaster end
        }

        List<double> bxOrthMean = new List<double>();
        
        public double[] __calcC(double[][] bxOrth)
        {
            var b = bxOrth[0].Length;
            double[] c = new double[b];
            double yAvg = Y.Average();

            for (int k = 0; k < bxOrth.Length; k++)
            {

                for (int i = 0; i < b; i++)
                {
                    c[i] += (Y[k] - yAvg) * bxOrth[k][i];
                }
            }
            return c;
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

        
        public double[][] __calcV(double[][] bxOrth)
        {
            var b = bxOrth[0].Length;
            double[][] v = new double[b][];
            for (int i = 0; i < b; i++)
                v[i] = new double[b];

            for (int k = 0; k < bxOrth.Length; k++)
            {
                for (int i = 0; i < b; i++)
                {
                    for (int j = 0; j < b; j++)
                    {
                        v[i][j] += bxOrth[k][j] * (bxOrth[k][i] - bxOrthMean[i]);
                    }
                }
            }
            return v;
        }

        public void CalcOrthColumn(ref double[][] bxOrth, double[] y, int newColumnAt)
        {
            int nTerms = newColumnAt;
            int nCases = bxOrth.Length;

            if (nTerms == 0)
            {
                double len = 1 / Math.Sqrt((double)nCases);
                for (int i = 0; i < nCases; i++)
                {
                    bxOrth[i][nTerms] = len;                     
                }
                bxOrthMean.Add(len);
            }
            else if (nTerms == 1)
            {
                double yMean = y.Average();                
                for (int i = 0; i < nCases; i++)
                    bxOrth[i][nTerms] = y[i] - yMean;
            }
            // resids go in rightmost col of bxOrth at nTerms
            else
            {
                //needfaster begin
                for (int p = 0; p < bxOrth.Length; p++)
                    bxOrth[p][newColumnAt] = y[p];
                //needfaster end
   
                for (int iTerm = 0; iTerm < nTerms; iTerm++)
                {
                    double Beta;
                    var pbxOrth = __getColumn(bxOrth, iTerm);
                    double xty = 0.0;
                    for (int i = 0; i < nCases; i++)
                        xty += pbxOrth[i] * y[i];
                    Beta = xty;

                    for (int i = 0; i < nCases; i++)
                        bxOrth[i][newColumnAt] -= Beta * pbxOrth[i];
                }
            }
            // normalize the column to length 1 and init bxOrthMean[nTerms]
            if (nTerms > 0)
            {
                double bxOrthSS = SumOfSquares(__getColumn(bxOrth, nTerms), nCases);
                
                if (bxOrthMean.Count - 1 != nTerms)
                    bxOrthMean.Add(0.0);

                bxOrthMean[nTerms] = __getColumn(bxOrth, nTerms).Average();

                double len = Math.Sqrt(bxOrthSS);
                for(int i = 0; i < nCases; i++)
                    bxOrth[i][nTerms] /= len;
            }
        }

        //-----------------------------------------------------------------------------
        // get mean centered sum of squares
        static double SumOfSquares(double [] x, double mean)
        {
            double ss = 0;
            //for(size_t i = 0; i < n; i++)
            for (int i = 0; i < x.Length; i++)
                ss += Math.Pow((x[i] - mean), 2.0);
            return ss;
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

        public double[] __initYcboSum(int nTerms, double [][] bxOrth, ref double pRssDeltaLin)
        {
            int nCases = Regressors.Length;
            //double[] CovCol = new double[m.RegressorsTransformed.Length + 1];
            //double[] CovSx = new double[m.RegressorsTransformed.Length + 1];
            //double[] ycboSum = new double[Model.MAX_TERMS_ALLOWED];
            double[] ycboSum = new double[Model.MAX_TERMS_ALLOWED];
            ycboSum[nTerms] = 0;
            var yMean = Y.Average();
            for (int i = 0; i < Regressors.Length; i++)
            {
                ycboSum[nTerms] += (Y[i] - yMean) * bxOrth[i][nTerms];
            }

            pRssDeltaLin = 0;
            double yboSum = 0;
            for(int i = 0; i < nCases; i++)
                yboSum += Y[i] * bxOrth[i][nTerms];
            
            pRssDeltaLin += Math.Pow(yboSum,2.0);

            return ycboSum;
        }


        internal int FindKnot(double[][] bxOrth, double[][] x, double[] ym, int nTerms, int iPred, int iParent
            , ref double[] CovCol, ref double[] CovSx, ref double [] ycboSum, ref double ybxSum)
        {
            int nCases = bxOrth.Length;
            var xOrder = this.GetArrayOrder(__getColumn(x, iPred));
            var bx = RegressorsTransformed;
            int iNewCol = nTerms;

            double bxSum = 0, bxSqSum = 0, bxSqxSum = 0, bxxSum = 0, st = 0;

            //double[] CovCol = new double[iNewCol + 1];
            //double[] CovSx = new double[iNewCol + 1];

            for (int i = nCases - 2; i >= 0; i--)
            {
                int ix0 = xOrder[i];   // get the x's in descending order
                double x0 = x[ix0][iPred];       // the knot (printed as Cut in trace prints)
                int ix1 = xOrder[i+1];
                double x1 = x[ix1][iPred];      // case next to the cut
                double bx1 = bx[ix1][iParent];
                double bxSq = Math.Pow(bx1,2.0);
                double xDelta = x1 - x0;          // will a lways be non negative

                for (int it = 0; it < iNewCol; it++)
                {
                    CovSx[it] += (bxOrth[ix1][it] - bxOrthMean[it]) * bx1;
                    CovCol[it] += xDelta * CovSx[it];
                }

                bxSum += bx1;
                bxSqSum += bxSq;
                bxxSum += bx1 * x1;
                bxSqxSum += bxSq * x1;
                double su = st;
                st = bxxSum - bxSum * x0;
                
                CovCol[iNewCol] += xDelta * (2 * bxSqxSum - bxSqSum * (x0 + x1)) +
                           (Math.Pow(su,2.0) - Math.Pow(st, 2.0)) / nCases;

                var yMean = this.Y.Average();
                //if (nResp == 1)
                {    // treat nResp==1 as a special case, for speed
                    ybxSum += (this.Y[ix1] - yMean) * bx1;
                    ycboSum[iNewCol] += xDelta * ybxSum;
                }
                

            }
                return 0;
        }
    }
}
