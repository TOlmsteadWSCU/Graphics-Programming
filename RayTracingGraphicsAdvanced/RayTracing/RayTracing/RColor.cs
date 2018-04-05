using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace RayTracing
{
    public class RColor
    {
        public double r, g, b;
        public RColor(double r, double g, double b)
        {
            this.r = r;
            this.g = g;
            this.b = b;
        }
        public RColor add(RColor c)
        {
            return new RColor(r + c.r, g + c.g, b + c.b);
        }
        public RColor scale(double d)
        {
            return new RColor(r * d, g * d, b * d);
        }
        public double intensity()
        {
            return Math.Max(r, Math.Max(g, b));
        }
        private int toi(double d)
        {
            return Math.Min(Math.Max((int)(d * 256), 0), 255);
        }
        public Color toColor()
        {
            // Console.WriteLine("r = " + r + " g = " + g + " b = " + b);
            return Color.FromArgb(toi(r), toi(g), toi(b));
        }
        public RColor mul(RColor c)
        {
            return new RColor(r * c.r, g * c.g, b * c.b);
        }
    }
}
