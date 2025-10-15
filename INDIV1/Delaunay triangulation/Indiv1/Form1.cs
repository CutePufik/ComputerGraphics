using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace DelaunayTriangulation
{
    public class Edge
    {
        public PointF P1 { get; }
        public PointF P2 { get; }

        public Edge(PointF p1, PointF p2)
        {
            P1 = p1;
            P2 = p2;
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Edge e)) return false;
            return (P1.Equals(e.P1) && P2.Equals(e.P2)) || (P1.Equals(e.P2) && P2.Equals(e.P1));
        }

        public override int GetHashCode()
        {
            return P1.GetHashCode() ^ P2.GetHashCode();
        }
    }

    public class Triangle
    {
        public PointF A { get; }
        public PointF B { get; }
        public PointF C { get; }

        public Triangle(PointF a, PointF b, PointF c)
        {
            A = a;
            B = b;
            C = c;
        }
    }
    public partial class Form1 : Form
    {
        private List<PointF> points = new List<PointF>();
        private List<Triangle> triangles = new List<Triangle>();
        private List<Edge> liveEdges = new List<Edge>();
        private Random rand = new Random();
        private bool isRunning = false;
        public Form1()
        {
            InitializeComponent();
            this.DoubleBuffered = true;
            timer = new System.Windows.Forms.Timer(components);
            timer.Interval = (int)delayNumericUpDown.Value;
            timer.Tick += new System.EventHandler(this.Timer_Tick);
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void generateButton_Click(object sender, EventArgs e)
        {
            int count = (int)numPointsNumericUpDown.Value;
            GeneratePoints(count);
        }

        private void startButton_Click(object sender, EventArgs e)
        {
            StartTriangulation();
        }

        private void clearButton_Click(object sender, EventArgs e)
        {
            points.Clear();
            triangles.Clear();
            liveEdges.Clear();
            isRunning = false;
            canvasPictureBox.Invalidate();
        }

        private void delayNumericUpDown_ValueChanged(object sender, EventArgs e)
        {
            timer.Interval = (int)delayNumericUpDown.Value;

        }

        private void numPointsNumericUpDown_ValueChanged(object sender, EventArgs e)
        {

        }

        private void canvasPictureBox_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.White);

            // Draw triangles
            using (var triPen = new Pen(Color.Blue, 1))
            {
                foreach (Triangle t in triangles)
                {
                    // ensure points valid
                    PointF[] pts = new PointF[] { t.A, t.B, t.C };

                    g.DrawPolygon(triPen, pts);
                }
            }

            // Draw live edges in red
            using (var edgePen = new Pen(Color.Red, 2))
            {
                foreach (Edge ed in liveEdges)
                {
                    g.DrawLine(edgePen, ed.P1, ed.P2);
                }
            }

            // Draw points on top
            using (var brush = new SolidBrush(Color.Black))
            using (var pen = new Pen(Color.White, 1))
            {
                foreach (PointF p in points)
                {
                    g.FillEllipse(brush, p.X - 1.5f, p.Y - 1.5f, 3, 3);
                    g.DrawEllipse(pen, p.X - 2, p.Y - 2, 4, 4);
                }
            }
        }

        private void canvasPictureBox_MouseClick(object sender, MouseEventArgs e)
        {
            if (!isRunning)
            {
                points.Add(new PointF(e.X, e.Y));
                canvasPictureBox.Invalidate();
            }
        }

        /// =============================================================///

        private void GeneratePoints(int count)
        {
            points.Clear();
            triangles.Clear();
            liveEdges.Clear();
            isRunning = false;

            int margin = 50;
            int w = canvasPictureBox.ClientSize.Width;
            int h = canvasPictureBox.ClientSize.Height;

            for (int i = 0; i < count; i++)
            {
                float x = (float)rand.NextDouble() * (w - 2 * margin) + margin;
                float y = (float)rand.NextDouble() * (h - 2 * margin) + margin;
                points.Add(new PointF(x, y));
            }

            canvasPictureBox?.Invalidate();
        }


        private void StartTriangulation()
        {
            if (points.Count < 3)
            {
                MessageBox.Show(
                    "Для построения триангуляции необходимо не менее трёх точек.",
                    "Недостаточно точек",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning
                );
                return;
            }
            triangles.Clear();
            liveEdges.Clear();
            isRunning = true;
            // Step 0: Compute convex hull using Jarvis (Gift Wrapping)
            List<PointF> hull = ComputeConvexHullJarvis(points);
            // Initialize with hull edges as live edges in reverse direction for correct orientation (CW, since hull is CCW)
            for (int i = 0; i < hull.Count; i++)
            {
                PointF p1 = hull[i];
                PointF p2 = hull[(i + 1) % hull.Count];
                liveEdges.Add(new Edge(p2, p1)); // Reverse to CW
            }

            timer.Start();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            if (!isRunning || liveEdges.Count == 0)
            {
                timer.Stop();
                isRunning = false;
                MessageBox.Show("Триангуляция завершена");
                return;
            }

            // Process the first live edge per step
            Edge currentEdge = liveEdges[0];
            ProcessLiveEdge(currentEdge);
            canvasPictureBox.Invalidate();
        }

        private void ProcessLiveEdge(Edge edge)
        {
            // Find right conjugate point
            PointF? conjugate = FindRightConjugate(edge, points);
            if (conjugate == null)
            {
                // No points to the right, it's a boundary edge, remove from live
                liveEdges.Remove(edge);
                return;
            }

            PointF c = conjugate.Value;
            // Create new triangle
            Triangle tri = new Triangle(edge.P1, edge.P2, c);
            triangles.Add(tri);

            // New edges: P1-C and P2-C, with correct orientation
            Edge e1 = CreateOrientedEdge(edge.P1, c, edge.P2);
            Edge e2 = CreateOrientedEdge(edge.P2, c, edge.P1);

            // Modify lists
            UpdateEdge(e1);
            UpdateEdge(e2);

            // Remove processed edge
            liveEdges.Remove(edge);
        }

        private Edge CreateOrientedEdge(PointF a, PointF b, PointF third)
        {
            float dx = b.X - a.X;
            float dy = b.Y - a.Y;
            float tx = third.X - a.X;
            float ty = third.Y - a.Y;
            float cross = dx * ty - dy * tx;
            if (cross < 0)
            {
                // Flip to make third on the left
                PointF temp = a;
                a = b;
                b = temp;
            }
            return new Edge(a, b);
        }

        private void UpdateEdge(Edge e)
        {
            // Check if already in live edges (undirected)
            Edge existing = liveEdges.Find(ex => ex.Equals(e));
            if (existing != null)
            {
                liveEdges.Remove(existing); // Becomes dead
            }
            else
            {
                liveEdges.Add(e);
            }
        }

        private PointF? FindRightConjugate(Edge edge, List<PointF> allPoints)
        {
            PointF? best = null;
            double minDist = double.MaxValue;

            foreach (PointF p in allPoints)
            {
                if (p.Equals(edge.P1) || p.Equals(edge.P2)) continue;
                if (!IsToTheRight(edge, p)) continue;

                // Compute circumcenter
                PointF center = Circumcenter(edge.P1, edge.P2, p);
                if (float.IsNaN(center.X)) continue; // Collinear

                // Check if the circumcircle is empty
                double R2 = DistanceSquared(center, edge.P1);
                bool empty = true;
                foreach (PointF q in allPoints)
                {
                    if (q.Equals(p) || q.Equals(edge.P1) || q.Equals(edge.P2)) continue;
                    if (DistanceSquared(center, q) < R2 - 1e-6)
                    {
                        empty = false;
                        break;
                    }
                }
                if (!empty) continue;

                // Use signed distance with flipped sign to match lecture (negative when center left)
                double dist = -SignedDistanceToLine(edge.P1, edge.P2, center);

                if (dist < minDist)
                {
                    minDist = dist;
                    best = p;
                }
            }
            return best;
        }

        private bool IsToTheRight(Edge edge, PointF p)
        {
            // Cross product to determine side
            float cross = (edge.P2.X - edge.P1.X) * (p.Y - edge.P1.Y) - (edge.P2.Y - edge.P1.Y) * (p.X - edge.P1.X);
            return cross < 0; // Right side
        }

        private double SignedDistanceToLine(PointF a, PointF b, PointF p)
        {
            // Signed distance from p to line AB
            double num = (b.X - a.X) * (p.Y - a.Y) - (b.Y - a.Y) * (p.X - a.X);
            double den = Math.Sqrt((b.X - a.X) * (b.X - a.X) + (b.Y - a.Y) * (b.Y - a.Y));
            return num / den;
        }

        private PointF Circumcenter(PointF a, PointF b, PointF c)
        {
            float d = 2 * (a.X * (b.Y - c.Y) + b.X * (c.Y - a.Y) + c.X * (a.Y - b.Y));
            if (Math.Abs(d) < 1e-6) return new PointF(float.NaN, float.NaN); // Collinear
            float ux = ((a.X * a.X + a.Y * a.Y) * (b.Y - c.Y) + (b.X * b.X + b.Y * b.Y) * (c.Y - a.Y) + (c.X * c.X + c.Y * c.Y) * (a.Y - b.Y)) / d;
            float uy = ((a.X * a.X + a.Y * a.Y) * (c.X - b.X) + (b.X * b.X + b.Y * b.Y) * (a.X - c.X) + (c.X * c.X + c.Y * c.Y) * (b.X - a.X)) / d;
            return new PointF(ux, uy);
        }

        private List<PointF> ComputeConvexHullJarvis(List<PointF> pts)
        {
            if (pts.Count < 3) return new List<PointF>(pts);

            // Find leftmost point
            PointF start = pts[0];
            foreach (PointF p in pts)
            {
                if (p.X < start.X || (p.X == start.X && p.Y < start.Y)) start = p;
            }

            List<PointF> hull = new List<PointF> { start };
            PointF current = start;
            PointF prevDirection = new PointF(0, -1); // Initial direction downward

            do
            {
                PointF next = pts[0];
                if (next.Equals(current)) next = pts[1];
                double minAngle = double.MaxValue;
                double maxDist = 0; // For ties, farthest
                foreach (PointF p in pts)
                {
                    if (p.Equals(current)) continue;
                    double angle = GetAngle(prevDirection, new PointF(p.X - current.X, p.Y - current.Y));
                    double dist = DistanceSquared(current, p);
                    if (angle < minAngle || (angle == minAngle && dist > maxDist))
                    {
                        minAngle = angle;
                        maxDist = dist;
                        next = p;
                    }
                }
                hull.Add(next);
                prevDirection = new PointF(next.X - current.X, next.Y - current.Y);
                current = next;
            } while (!current.Equals(start));

            hull.RemoveAt(hull.Count - 1); // Remove duplicate start
            return hull;
        }

        private double GetAngle(PointF v1, PointF v2)
        {
            double dot = v1.X * v2.X + v1.Y * v2.Y;
            double det = v1.X * v2.Y - v1.Y * v2.X;
            double angle = Math.Atan2(det, dot);
            if (angle < 0) angle += 2 * Math.PI;
            return angle;
        }

        private float DistanceSquared(PointF a, PointF b)
        {
            float dx = a.X - b.X;
            float dy = a.Y - b.Y;
            return dx * dx + dy * dy;
        }

        private void Form1_Paint(object sender, PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            // Draw points
            foreach (PointF p in points)
            {
                g.FillEllipse(Brushes.Black, p.X - 2, p.Y - 2, 4, 4);
            }
            // Draw triangles
            foreach (Triangle t in triangles)
            {
                g.DrawPolygon(Pens.Blue, new PointF[] { t.A, t.B, t.C });
            }
            // Draw live edges in red
            foreach (Edge ed in liveEdges)
            {
                g.DrawLine(Pens.Red, ed.P1, ed.P2);
            }
        }

        private void Form1_MouseClick(object sender, MouseEventArgs e)
        {
            if (!isRunning)
            {
                points.Add(new PointF(e.X, e.Y));
                Invalidate();
            }
        }

        private void delayLabel_Click(object sender, EventArgs e)
        {

        }
    }
}
