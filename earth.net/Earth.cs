﻿using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Data;
using System.IO;
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

        Model m ;

        public List<double> Predict(string value)
        {
            m = new Model(GetX(value), GetColumnDouble("mpg").ToArray());
            
            //hinge test
            var xs = GetX("mpg");

            
            int MAX_HINGES_IN_BASIS = 30;
            int MAX_BASISES = 15;
            double MAX_DELTA_RSS = 0.00000001;
            int DATASET_ROWS = m.Y.Length;

            //B0
            m.AddBasis(new Basis(null, null, 1.0, DATASET_ROWS), null);

            do
            {
                int solutions = 0;    

                for (int i = 0; i < m.Basises.Count; i++)
                {
                    double PotentialRSS = m.RSS;

                    bool betterFound = false;

                    int varN = 0;
                    int valN = 0;

                    double[][] bData;

                    //There is one restriction put on the formation of model terms: each input
                    //can appear at most once in a product.
                    for (int j = 0; j < m.Regressors[0].Length; j++)
                    {
                        if (m.Basises[i].IsInputAppearsInProduct(j))
                            continue;

                        if (m.Basises[i].HingesCount >= MAX_HINGES_IN_BASIS)
                            break;

                        Hinge h = new Hinge(j, 0.0);
                        Hinge hReflected = h.ConstructNegative();

                        Basis b = new Basis(m.Basises[i], h, DATASET_ROWS);
                        Basis bReflected = new Basis(m.Basises[i], hReflected, DATASET_ROWS);

                        bData = null;
                        
                        int[] kOrdered = m.GetArrayOrder(m.GetRegressorsColumn(j));
                        m.uValue1 = double.MinValue;
                        m.uValue2 = double.MinValue;
                        for (int k = m.Regressors.Length - 2; k >= 0; k--)
                        {
                             //каков индекс в регрессорах k по убыванию элемента
                             int ki0 = kOrdered[k];
                             int ki1 = kOrdered[k + 1]   ; //kOrdered[ki1] должно быть больше или равно нулевого, у нас же убывающий порядок
                             
                             double k0 = m.Regressors[ki0][j];
                             double k1 = m.Regressors[ki1][j];
 
                             double kdiff = k1 - k0;
                             
                             if (kdiff < 0)
                                 throw new Exception("t should be <= u !");
                            //k0  -- t
                            //k1  -- u
                            h.Value = k0;
                            hReflected.Value = k0;

                            //double rss = m.CheckNewBasisCholessky(b, bReflected);
                            double rss = m.CheckNewBasisCholeskyFast(b, bReflected, k1, ref bData);
                            //double rss = m.CheckNewBasisEquation52(b, bReflected, 0.0, ref bData);
                            if (rss == double.MaxValue)
                                continue;
                            Console.WriteLine("Cholessky rss = " + rss);
                            // rss = m.CheckNewBasisFast(b, bReflected, 0.0, ref bData);
                             //Console.WriteLine("Fast rss = " + rss);
                            //double rss = m.CheckNewBasis(b, bReflected);
                            
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
                        solutions++;
                        
                        Hinge winnerHinge = new Hinge(varN, m.Regressors[valN][varN]);
                        Hinge winnerHingeReflected = winnerHinge.ConstructNegative();

                        Basis winnerBasis = new Basis(m.Basises[i], winnerHinge, DATASET_ROWS);
                        Basis winnerBasisReflected = new Basis(m.Basises[i], winnerHingeReflected, DATASET_ROWS);

                        m.AddBasis(winnerBasis, winnerBasisReflected);
                        if (m.Basises.Count >= MAX_BASISES)
                            break;
                    }
                }
                

                if (solutions == 0) break; //no solutions anymore which decrease RSS
                if (m.Basises.Count >= MAX_BASISES) break; 
                if (m.Basises.Any(b => b.HingesCount > MAX_HINGES_IN_BASIS)) break;
            }
            while (true);


            //Pruning pass
            
            
            double [] GCVs = new double[m.Basises.Count];

            using (System.IO.StreamWriter t = new StreamWriter("output.txt"))
            {

                t.WriteLine(RegressionToolkit.DoubleToR(m.RegressorsTransformed));
                t.WriteLine(RegressionToolkit.DoubleToR(m.Y));
            }

            do
            {

                double lowestGCV = 1000000.0;
                int lowestGCVIndex = 1;
                Basis[] tempBasises = new Basis[m.Basises.Count];
                m.Basises.CopyTo(tempBasises);

                for (int i = 1; i < m.Basises.Count; i++)
                {
                    m.RemoveBasisAt(i);
                    if (m.GCV < lowestGCV)
                    {
                        lowestGCV = m.GCV;
                        lowestGCVIndex = i;
                    }
                    m.ResetBasis(tempBasises);
                    //using (System.IO.StreamWriter t = new StreamWriter("output.txt"))
                    //{
                    //    t.WriteLine(RegressionToolkit.DoubleToR(m.RegressorsTransformed));
                    //}
                }
                m.RemoveBasisAt(lowestGCVIndex);
                
                if (m.Basises.Count == 3)
                    break;
            }
            while (true);

            Console.WriteLine(Model._warns);
            Console.WriteLine(RegressionToolkit._bad);
            Console.WriteLine(RegressionToolkit._good);
                return new List<double>();
        }
    }
}
