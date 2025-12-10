namespace cornish_room;

public class Camera
{
    public Point3D Position { get; set; }
    public float Fov { get; set; }

    public Camera(Point3D position, float fov)
    {
        Position = position;
        Fov = fov;
    }
}
