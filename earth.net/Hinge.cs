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
        public bool Negative = false;
        
        public double Calc(double x)
        {
            return (Negative? -1 : +1) *( x - Value );
        }

        public Hinge ConstructNegative()
        {
            Hinge h = new Hinge();
            h.Value = this.Value;
            h.Variable = this.Variable;
            h.Negative = !this.Negative;
            return h;
        }

        public Hinge()
        {
        }

        public Hinge(int variable, double value)
        {
            this.Variable = variable;
            this.Value = value;
            this.Negative = false;
        }

        public override string ToString()
        {
            if(Negative)
                return string.Format("{1} - X{0}", Variable, Value);
            else
            return string.Format("X{0} - {1}", Variable, Value);
        }
    }
}
