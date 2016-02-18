using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    public class Earth
    {
        DataTable _dt;

        List<string> _variables = null;
        public List<string> Variables
        {
            get
            {
                if (_variables == null)
                    _variables = (from DataColumn dc in _dt.Columns
                                  select dc.ColumnName).ToList();
                return _variables;
            }
        }

        double[][] _values = null;

        /// <summary>
        /// Переписать нахуй эту корягу
        /// </summary>
        /// <param name="y"></param>
        /// <returns></returns>
        public List<string> GetRegressors(string y)
        {
            if (_variables == null)
                _variables = (from DataColumn dc in _dt.Columns
                              where (dc.DataType == typeof(double) || dc.DataType == typeof(int)) 
                                && dc.ColumnName != y
                              select dc.ColumnName).ToList();
            return _variables;
        }

        public double[][] Values
        {
            get
            {
                if (_values == null) _values = GetX("");
                return _values;
            }
        }

        /// <summary>
        /// Вернуть кололнку типа double
        /// </summary>
        /// <param name="colname"></param>
        /// <returns></returns>
        private List<double> GetColumnDouble(string colname)
        {

            if (_dt.Columns[colname].DataType != typeof(double))
                return null;

            var vals = (from DataRow dr in _dt.Rows
                        select (double)dr[colname]).ToList();

            return vals;
        }

        /// <summary>
        /// Вернуть двумерный массив регрессоров (независимых переменных), без зависимой переменной (y)
        /// </summary>
        /// <param name="ycolname"></param>
        /// <returns></returns>
        public double[][] GetX(string ycolname)
        {
            var XColnames = (from dc in _dt.Columns.Cast<DataColumn>()
                             where (dc.DataType == typeof(double) || dc.DataType == typeof(int))
                             select dc.ColumnName).ToList();
            XColnames.Remove(ycolname);
            double[][] result = new double[_dt.Rows.Count][];

            for (int i = 0; i < _dt.Rows.Count; i++)
            {
                result[i] = new double[XColnames.Count];
                for (int j = 0; j < XColnames.Count; j++)
                {
                    object v = _dt.Rows[i][XColnames[j]];

                    result[i][j] = v is double ? (double)v : (double)((int)v);
                }
            }

            return result;
        }

        public Earth(DataTable dt)
        {
            _dt = dt;
        }

        public List<double> CalculateLeastSquares(string ycolName)
        {
            var x = GetX(ycolName);
            var y = GetColumnDouble(ycolName).ToArray();

            return RegressionToolkit.CalculateLeastSquares(x, y);
        }

        

        
      

        Model m ;

        public List<double> Predict(string value)
        {
            m = new Model(GetX(value), GetColumnDouble("mpg").ToArray());
            
            //var c = CalculateLeastSquares("mpg").ToArray();
            //var predicted = RegressionToolkit.Predict(c, GetX(value));
            //var mpg = GetColumnDouble("mpg");
            //var RSS = RegressionToolkit.calcRSS(predicted.ToArray(), mpg.ToArray());
            //Console.WriteLine(RSS);

            //hinge test
            var xs = GetX("mpg");

            //Hinge h = new Hinge(0, 3.7);
            //var b0 = new Basis(null, null, 1.0);
            
            //var b1 = new Basis(b0, h);
            //var b1Reflected = new Basis(b0, h.ConstructNegative());
            

            //m.AddBasis(b0, null);
            //m.AddBasis(b1, b1Reflected);

            
            int MaxTerms = 100;
            //B0
            m.AddBasis(new Basis(null, null, 1.0), null);
            do
            {
                for (int i = 0; i < m.Basises.Count; i++)
                {
                    double PotentialRSS = m.RSS;

                    bool betterFound = false;

                    int varN = 0;
                    int valN = 0;


                    //There is one restriction put on the formation of model terms: each input
                    //can appear at most once in a product.
                    for (int j = 0; j < GetRegressors(value).Count; j++)
                    {
                        if (m.Basises[i].IsInputAppearsInProduct(j))
                            continue;

                        if (m.Basises[i].TermsCount >= MaxTerms)
                            break;
                        
                        for (int k = 0; k < Values.Length; k++)
                        {
                            Hinge h = new Hinge(j, Values[k][j]);
                            Hinge hReflected = h.ConstructNegative();

                            Basis b = new Basis(m.Basises[i], h);
                            Basis bReflected = new Basis(m.Basises[i], hReflected);

                            double rss = m.CheckNewBasis(b, bReflected);
                            if (rss < PotentialRSS)
                            {
                                PotentialRSS = rss;
                                varN = j;
                                valN = k;
                                betterFound = true;
                            }
                        }
                    }

                    if (betterFound)
                    {
                        Hinge winnerHinge = new Hinge(varN, Values[valN][varN]);
                        Hinge winnerHingeReflected = winnerHinge.ConstructNegative();

                        Basis winnerBasis = new Basis(m.Basises[i], winnerHinge);
                        Basis winnerBasisReflected = new Basis(m.Basises[i], winnerHingeReflected);

                        m.AddBasis(winnerBasis, winnerBasisReflected);
                    }
                }
            }
            while (true);

            return new List<double>();
        }
    }
}
