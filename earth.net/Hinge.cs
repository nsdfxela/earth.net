using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace earth.net
{
    /// <summary>
    /// Хиндж (он же "хоккейная клюшка")
    /// </summary>
    public class Hinge
    {
        public int? Variable;
        public double Value;
        public double Calc(double x)
        {
            return x - Value;
        }

        public override string ToString()
        {
            return string.Format("X{0} - {1}", Variable, Value);
        }
    }
}
