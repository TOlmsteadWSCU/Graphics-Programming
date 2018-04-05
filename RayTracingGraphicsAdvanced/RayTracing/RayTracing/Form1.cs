using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace RayTracing
{
    public partial class Form1 : Form
    {
        public List<SceneObject> scenes = new List<SceneObject>();
        public Form1()
        {
            InitializeComponent();
            traceRays();
            Ray r = new Ray(cameraPosition, new P3(0.0, 1.0, 0));
            Console.WriteLine(sphere.intersect(r));
        }
        //public void shootRay(Ray r, out double dist)
        //{

        //}
        public static P3 cameraPosition = new P3(0.0, -5.0, 0.0);
        public const int pixels = 100;
        public static RColor[,] image = new RColor[pixels, pixels];
        public Sphere sphere = new Sphere(new P3(0.0, 0.0, 0.0), .8f);

        public void traceRays()
        {
            for (int i = 0; i < pixels; i++)
            {
                for (int j = 0; j < pixels; j++)
                {
                    P3 projPoint = new P3(2 * (((double)i) / pixels - 0.5), 0, 2 * (((double)j) / pixels - 0.5));
                    P3 dir = projPoint.sub(cameraPosition).norm();
                    Ray r = new Ray(cameraPosition, dir);
                    float dist = sphere.intersect(r);
                    //Console.WriteLine(dist + " " + i + " " + j);
                    // SceneObject o = shootRay(r, out dist);  // You need to write shootRay.  -1 indicates that no object was intersected.
                    if (dist == -1)
                    {
                        image[i, j] = new RColor(0, 0, 0);
                    }
                    else
                    {
                        P3 p = r.travel(dist);
                        P3 normal = sphere.normal(p);
                        double k = normal.dot(new P3(-.5, -.5, .5).norm());
                        RColor intensity = new RColor(.3 + .7 * Math.Max(0, k), 0, 0);
                        image[i, j] = intensity;
                    }
                }
            }
        }//tracerays
        public SceneObject shootRay(Ray r, SceneObject o)
        {
            SceneObject c = null;
            float dist = -1;
            foreach(SceneObject s in scenes)
            {
                if(s != o)
                {
                    dist = (float)s.distance(r);
                }
            }
        }
        private void panel1_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;

            for (int i = 0; i < pixels; i++)
            {
                for (int j = 0; j < pixels; j++)
                {
                    Color c = image[i, j].toColor();
                    Pen pen = new Pen(c);
                    g.FillRectangle(pen.Brush, i * 2, j * 2, 2, 2);
                }
            }
        }
    }
}
