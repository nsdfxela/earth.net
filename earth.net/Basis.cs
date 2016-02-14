using System;
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
        public int TermsCount
        {
            get { return Hinges.Count; }
        }
        public Basis(Basis parent, Hinge hinge)
        {
            if (parent != null)
                Hinges.AddRange(parent.Hinges);

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

        public Basis(Basis parent, int? variable, double value) : this(parent, new Hinge { Value = value, Variable = variable }) { }

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

        public List<Hinge> Hinges = new List<Hinge>();
    }

}
