namespace cornish_room;

public class Ray
{
    public Point3D Origin { get; set; }
    public Point3D Direction { get; set; }

    public Ray(Point3D origin, Point3D direction)
    {
        Origin = origin;
        Direction = direction;
    }
}