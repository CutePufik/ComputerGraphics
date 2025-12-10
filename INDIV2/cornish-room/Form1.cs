namespace cornish_room
{
    public partial class Form1 : Form
    {
        private Graphics _graphics;
        private Bitmap _bitmap;
        private Camera _camera;
        private Scene _scene;
        private Renderer _renderer;

        private Point3D _cameraPosition = new Point3D(0, 0, -10);

        public Form1()
        {
            InitializeComponent();
            InitGraphics();
            InitRenderer();
        }

        private void InitGraphics()
        {
            _bitmap = new Bitmap(pictureBoxScene.Width, pictureBoxScene.Height);
            _graphics = Graphics.FromImage(_bitmap);
            _graphics.Clear(Color.GhostWhite);
            pictureBoxScene.Image = _bitmap;
        }

        private void InitRenderer()
        {
            _scene = new Scene();
            _camera = new Camera(_cameraPosition, (float)(Math.PI / 3 * 2));
            InitShapes();
            InitLights();
            _renderer = new Renderer(pictureBoxScene.Width, pictureBoxScene.Height, _camera, _scene);
        }

        private void InitShapes()
        {
            List<Point3D> points = new List<Point3D>
            {
                new Point3D(-10, -10, 10), 
                new Point3D(-10, -10, -10), 
                new Point3D(-10, 10, -10), 
                new Point3D(-10, 10, 10), 
                new Point3D(10, -10, 10),
                new Point3D(10, -10, -10),
                new Point3D(10, 10, -10), 
                new Point3D(10, 10, 10) 
            };
            
            Face frontWall = new Face(
                new List<Point3D> { points[0], points[3], points[7], points[4] },
                color: Color.LightPink,
                specularity: radioButtonFrontWallSpecularity.Checked,
                transparency: radioButtonFrontWallTransparency.Checked
            );

            Face backWall = new Face(
                new List<Point3D> { points[1], points[5], points[6], points[2] },
                color: Color.LightSalmon,
                specularity: radioButtonBackWallSpecularity.Checked,
                transparency: radioButtonBackWallTransparency.Checked
            );
            
            Face upWall = new Face(
                new List<Point3D> { points[3], points[2], points[6], points[7] },
                color: Color.LightGreen,
                specularity: radioButtonUpWallSpecularity.Checked,
                transparency: radioButtonUpWallTransparency.Checked
            );

            Face downWall = new Face(
                new List<Point3D> { points[0], points[4], points[5], points[1] },
                color: Color.LightGoldenrodYellow,
                specularity: radioButtonDownWallSpecularity.Checked,
                transparency: radioButtonDownWallTransparency.Checked
            );
            
            Face rightWall = new Face(
                new List<Point3D> { points[4], points[7], points[6], points[5] },
                color: Color.LightSkyBlue,
                specularity: radioButtonRightWallSpecularity.Checked,
                transparency: radioButtonRightWallTransparency.Checked
            );

            Face leftWall = new Face(
                new List<Point3D> { points[0], points[1], points[2], points[3] },
                color: Color.LightCoral,
                specularity: radioButtonLeftWallSpecularity.Checked,
                transparency: radioButtonlLeftWallTransparency.Checked
            );
            
            Sphere sphere1 = new Sphere(
                new Point3D(-6, -3.5f, 5),
                radius: 2,
                color: Color.Magenta,
                specularity: radioButtonSphere1Specularity.Checked,
                transparency: radioButtonSphere1Transparency.Checked);
            
            Sphere sphere2 = new Sphere(
                new Point3D(4, -2.5f, 5),
                radius: 3,
                color: Color.Crimson,
                specularity: radioButtonSphere2Specularity.Checked,
                transparency: radioButtonSphere2Transparency.Checked);
            
            Cube cube1 = new Cube(
                new Point3D(-6, -7.5f, 5), 
                3, 5, 3,
                color: Color.Aqua,
                specularity: radioButtonCube1Specularity.Checked,
                transparency: radioButtonCube1Transparency.Checked);
            
            Cube cube2 = new Cube(
                new Point3D(4, -7.5f, 5), 
                5, 5, 5, 
                color: Color.Gold, 
                specularity: radioButtonCube2Specularity.Checked,
                transparency: radioButtonCube2Transparency.Checked);
            
            _scene.Shapes.Clear();
            
            _scene.AddShape(frontWall);
            _scene.AddShape(backWall);
            _scene.AddShape(upWall);
            _scene.AddShape(downWall);
            _scene.AddShape(rightWall);
            _scene.AddShape(leftWall);

            _scene.AddShape(sphere1);
            _scene.AddShape(sphere2);
            _scene.AddShape(cube1);
            _scene.AddShape(cube2);
        }

        private void InitLights()
        {
            _scene.Lights.Clear();
            
            Light light1 = new Light(new Point3D(0, 9, 0), 1.0f);
            _scene.AddLight(light1);
            
            if (checkBoxSecondLight.Checked)
            {
                Light light2 = new Light(
                    new Point3D(
                        (float)numericUpDownX.Value,
                        (float)numericUpDownY.Value,
                        (float)numericUpDownZ.Value),
                    intensity: 1.0f);
                
                _scene.AddLight(light2);
            }
        }
        
        private void buttonStartRender_Click(object sender, EventArgs e)
        {
            RenderScene();
        }

        private void RenderScene()
        {
            InitShapes();
            InitLights();
            _renderer = new Renderer(pictureBoxScene.Width, pictureBoxScene.Height, _camera, _scene);
            _renderer.Render(_graphics);
            pictureBoxScene.Refresh();
        }
    }
}
