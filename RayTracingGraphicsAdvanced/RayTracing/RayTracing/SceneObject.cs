using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RayTracing
{
    public class SceneObject
    {
        public Geometry g;
        public Skin s;
        public Texture t;
        public SceneObject(Geometry g, Texture t, Skin s)
        {
            this.g = g;
            this.s = s;
            this.t = t;
        }
        public double distance(Ray r)
        {
            return g.intersect(r);
        }
        public RColor color(P3 p, Ray r)
        {
            return s.color(p, r, this);
        }
    }
}
