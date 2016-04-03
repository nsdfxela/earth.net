using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    /// <summary>
    /// Базисная функция. Произведение хинджей, составляющая модели
    /// </summary>
    public class Basis
    {
        public Basis Parent;

        public int HingesCount
        {
            get { return Hinges.Count; }
        }

        public Basis(Basis parent, Hinge hinge, int DataSetRows)
        {
            if (parent != null)
                Hinges.AddRange(parent.Hinges);
            
            Parent = parent;
            ht = new double[DataSetRows];
            for (int h = 0; h < ht.Length; h++)
                ht[h] = 1.0;
            htExists = new bool[DataSetRows];
            
            Hinges.Add(hinge);
        }

        public Basis(List<Hinge> parentHinges, Hinge hinge)
        {
            Hinges.AddRange(parentHinges);
            Hinges.Add(hinge);
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            foreach (var h in Hinges)
            {
                sb.Append(string.Format("({0})*", h.ToString()));
            }
            return sb.ToString().TrimEnd(new char[] { '*' });
        }

        public Basis(Basis parent, int? variable, double value, int dataSetRows) : this(parent, new Hinge { Value = value, Variable = variable }, dataSetRows) { }

        public double Calc(double[] x)
        {
            double result = 1.0;
            for (int i = 0; i < Hinges.Count; i++)
            {
                int? xn = Hinges[i].Variable;
                if (xn != null)
                    result *= Hinges[i].Calc(x[(int)xn]);
            }
            return result;
        }


        //public Hashtable ht = new Hashtable();
        double [] ht;
        bool[] htExists;

        private bool _htExists(int v)
        {
            return htExists[v];
        }
        private void _addToHt(int i, double val)
        {
            ht[i] = val;
            htExists[i] = true;
        }

        public double CalcFast(double []x, int hash)
        {
            
            double parentResult;
            if (_htExists(hash))
                parentResult = ht[hash];
            else if (Parent == null)
                return 1.0;
            else
            {
                parentResult = Parent.CalcFast(x, hash);
                _addToHt(hash, parentResult);
            }

            double result = 0.0;
            int iActualHinge = Hinges.Count - 1;
            int? xn = Hinges[iActualHinge].Variable;
            if (xn != null)
                result = parentResult * Hinges[iActualHinge].Calc(x[(int)xn]);
            return result;
        }

        public List<Hinge> Hinges = new List<Hinge>();

        /// <summary>
        /// There is one restriction put on the formation of model terms: each input
        ///can appear at most once in a product.
        /// </summary>
        /// <param name="variableNumber">Number of variable</param>
        /// <returns></returns>
        internal bool IsInputAppearsInProduct(int variableNumber)
        {
            return Hinges.Any(h => h.Variable == variableNumber);
        }
    }

    public class HingePair
    {
        Hinge Positive { get; set; }
        Hinge Negative { get; set; }
        
        public HingePair(Hinge positive) : this(positive, positive.ConstructNegative()){ }

        public HingePair(Hinge positive, Hinge negative)
        {
            Positive = positive;
            Negative = negative;
        }
    }

}
