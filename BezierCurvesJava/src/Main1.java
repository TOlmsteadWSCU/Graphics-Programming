
import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Owner
 */
public class Main1 {
    public Point p = new Point(0,0);
    public static ArrayList<Point> points = new ArrayList<>();
    public static double t;
    
    public static void main(String[] args)
    {
        
        points.add(new Point(0, 0));
        points.add(new Point(0, 100));
        points.add(new Point(100, 100));
        points.add(new Point(100, 0));
        Bspline b = new Bspline(points, true);
        Bezier b1 = new Bezier(points);
        //Point p3 = b.eval(0);

        
            System.out.println("Bezier: " + b1);
           
            System.out.println(b.bez[0]);
      
        

    }
}
