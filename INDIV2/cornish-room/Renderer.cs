namespace cornish_room
{
    public class Renderer
    {
        private const int MaxDepth = 10;
        private int _width, _height;
        private Camera _camera;
        private Scene _scene;
        private Color[,] _colorsBuffer; // двумерный массив цветов, где хранитс€ результат дл€ каждого пиксел€.

        public Renderer(int width, int height, Camera camera, Scene scene)
        {
            _width = width;
            _height = height;
            _camera = camera;
            _scene = scene;
            _colorsBuffer = new Color[width, height];
        }

        public void Render(Graphics g)
        {
            for (int i = 0; i < _width; ++i)
            {
                for (int j = 0; j < _height; ++j)
                {
                    float x = (2 * (i + 0.5f) / _width - 1) * (float)Math.Tan(_camera.Fov / 2) * (_width / (float)_height);
                    float y = -(2 * (j + 0.5f) / _height - 1) * (float)Math.Tan(_camera.Fov / 2);
                    Ray ray = new Ray(_camera.Position, new Point3D(x, y, _camera.Fov).Normalize());
                    Color color = Trace(ray, MaxDepth, null);
                    _colorsBuffer[i, j] = color;
                }
            }
            PaintScene(g);
        }

        private void PaintScene(Graphics g)
        {
            for (int i = 0; i < _width; ++i)
            {
                for (int j = 0; j < _height; ++j)
                {
                    var color = _colorsBuffer[i, j];
                    g.FillRectangle(new SolidBrush(color), i, j, 1, 1);
                }
            }
        }

        private (bool collide, float dist, Point3D point, Point3D normal, Shape shape) SceneCollide(Ray ray, Shape currentObj)
        {
            // поиск ближайшего пересечени€
            var shapes = _scene.Shapes
                .Where(shape => shape != currentObj)
                .Select(shape => shape.Collision(ray))
                .ToList();

            var nearestShape = shapes
                .Where(result => result.collide)
                .OrderBy(result => result.dist)
                .FirstOrDefault();

            return nearestShape;
        }

        private Color Trace(Ray ray, int depth, Shape currentObj)
        {
            if (depth <= 0) 
                return _scene.BackgroundColor;

            var result = SceneCollide(ray, currentObj);
            var point = result.point;
            var normal = result.normal;

            // если ни во что не попал Ч фон
            if (!result.collide) 
                return _scene.BackgroundColor;

            // вычисл€ем освещение
            float lightIntensity = 0;
            foreach (var light in _scene.Lights)
            {
                var lightDir = (light.Position - point).Normalize();  //  направление от точки к источнику света
                var lightDistance = (light.Position - point).Length(); // рассто€ние от точки до источника света
                Ray shadowRay = new Ray(point, lightDir);

                if (SceneCollide(shadowRay, result.shape).dist < lightDistance)
                {
                    // точка в тени
                    lightIntensity += 0.3f / _scene.Lights.Count * light.Intensity * Math.Max(0, lightDir.DotProduct(normal));
                    continue;
                }
                // точка не в тени
                lightIntensity += light.Intensity * Math.Max(0, lightDir.DotProduct(normal));
            }
            // усредн€ем освещенность по всем источникам света
            lightIntensity /= _scene.Lights.Count;

            // проверить отражени€ (если объект зеркальный)
            Color reflection = Color.Empty;
            if (result.shape.Specularity)
            {
                var reflectedRay = Reflect(point, ray, normal);
                reflection = Trace(reflectedRay, depth - 1, result.shape);
            }

            // проверить прозрачность (если объект прозрачный)
            Color refraction = Color.Empty;
            if (result.shape.Transparency)
            {
                var refractedRay = Refract(point, ray, normal);
                refraction = Trace(refractedRay, depth - 1, result.shape);
            }
            // смешать все цвета
            Color color = result.shape.Color;
            color = color.Multiply(lightIntensity);

            if (result.shape.Specularity)
                color = color.Add(reflection.Multiply(depth / (float)MaxDepth / 2));

            if (result.shape.Transparency)
                color = color.Add(refraction.Multiply(depth / (float)MaxDepth / 2));

            return color;
        }

        // ‘ункци€ вычислени€ отраженного луча (зеркальное отражение) R = D - 2(DЈN)N
        private Ray Reflect(Point3D point, Ray ray, Point3D normal)
        {
            var newDirection = ray.Direction - normal * 2 * ray.Direction.DotProduct(normal);
            return new Ray(point, newDirection);
        }

        // ‘ункци€ вычислени€ преломленного луча (закон —неллиуса)
        private Ray Refract(Point3D point, Ray ray, Point3D normal)
        {
            float product = -Math.Max(-1, Math.Min(1, ray.Direction.DotProduct(normal)));
            float eta1 = 1;
            float eta2 = 0.9f;
            
            Point3D norm = normal.Clone();
            
            if (product < 0)
            {
                product = -product;
                (eta2, eta1) = (eta1, eta2);
                norm *= -1;
            }

            float eta = eta1 / eta2;
            float cos = 1 - eta * eta * (1 - product * product);
            var newDirection = ray.Direction * eta + norm * (float)(eta * product - Math.Sqrt(cos));

            return new Ray(point, newDirection);
        }
    }
}
