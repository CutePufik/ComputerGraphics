namespace cornish_room;

public class Scene
{
    public List<Shape> Shapes { get; set; }
    public List<Light> Lights { get; set; }
    public Color BackgroundColor { get; set; }

    public Scene()
    {
        Shapes = new List<Shape>();
        Lights = new List<Light>();
        BackgroundColor = Color.Gray;
    }

    public void AddShape(Shape shape)
    {
        Shapes.Add(shape);
    }

    public void AddLight(Light light)
    {
        Lights.Add(light);
    }
}