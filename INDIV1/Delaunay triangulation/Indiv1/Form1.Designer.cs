namespace DelaunayTriangulation
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            numPointsNumericUpDown = new NumericUpDown();
            generateButton = new Button();
            delayLabel = new Label();
            canvasPictureBox = new PictureBox();
            startButton = new Button();
            clearButton = new Button();
            contextMenuStrip1 = new ContextMenuStrip(components);
            delayNumericUpDown = new NumericUpDown();
            panel1 = new Panel();
            ((System.ComponentModel.ISupportInitialize)numPointsNumericUpDown).BeginInit();
            ((System.ComponentModel.ISupportInitialize)canvasPictureBox).BeginInit();
            ((System.ComponentModel.ISupportInitialize)delayNumericUpDown).BeginInit();
            panel1.SuspendLayout();
            SuspendLayout();
            // 
            // numPointsNumericUpDown
            // 
            numPointsNumericUpDown.Location = new Point(12, 12);
            numPointsNumericUpDown.Maximum = new decimal(new int[] { 1000, 0, 0, 0 });
            numPointsNumericUpDown.Name = "numPointsNumericUpDown";
            numPointsNumericUpDown.Size = new Size(150, 27);
            numPointsNumericUpDown.TabIndex = 0;
            numPointsNumericUpDown.Value = new decimal(new int[] { 50, 0, 0, 0 });
            numPointsNumericUpDown.ValueChanged += numPointsNumericUpDown_ValueChanged;
            // 
            // generateButton
            // 
            generateButton.Location = new Point(168, 10);
            generateButton.Name = "generateButton";
            generateButton.Size = new Size(94, 29);
            generateButton.TabIndex = 1;
            generateButton.Text = "Generate";
            generateButton.UseVisualStyleBackColor = true;
            generateButton.Click += generateButton_Click;
            // 
            // delayLabel
            // 
            delayLabel.AutoSize = true;
            delayLabel.Location = new Point(300, 14);
            delayLabel.Name = "delayLabel";
            delayLabel.Size = new Size(83, 20);
            delayLabel.TabIndex = 2;
            delayLabel.Text = "Delay (ms):";
            // 
            // canvasPictureBox
            // 
            canvasPictureBox.BackgroundImageLayout = ImageLayout.Stretch;
            canvasPictureBox.BorderStyle = BorderStyle.FixedSingle;
            canvasPictureBox.Dock = DockStyle.Fill;
            canvasPictureBox.Location = new Point(0, 0);
            canvasPictureBox.Name = "canvasPictureBox";
            canvasPictureBox.Size = new Size(862, 518);
            canvasPictureBox.TabIndex = 4;
            canvasPictureBox.TabStop = false;
            canvasPictureBox.Paint += canvasPictureBox_Paint;
            canvasPictureBox.MouseClick += canvasPictureBox_MouseClick;
            // 
            // startButton
            // 
            startButton.Location = new Point(545, 10);
            startButton.Name = "startButton";
            startButton.Size = new Size(94, 29);
            startButton.TabIndex = 5;
            startButton.Text = "Start";
            startButton.UseVisualStyleBackColor = true;
            startButton.Click += startButton_Click;
            // 
            // clearButton
            // 
            clearButton.Location = new Point(645, 10);
            clearButton.Name = "clearButton";
            clearButton.Size = new Size(94, 29);
            clearButton.TabIndex = 6;
            clearButton.Text = "Clear";
            clearButton.UseVisualStyleBackColor = true;
            clearButton.Click += clearButton_Click;
            // 
            // contextMenuStrip1
            // 
            contextMenuStrip1.ImageScalingSize = new Size(20, 20);
            contextMenuStrip1.Name = "contextMenuStrip1";
            contextMenuStrip1.Size = new Size(61, 4);
            // 
            // delayNumericUpDown
            // 
            delayNumericUpDown.Location = new Point(389, 12);
            delayNumericUpDown.Maximum = new decimal(new int[] { 10000, 0, 0, 0 });
            delayNumericUpDown.Name = "delayNumericUpDown";
            delayNumericUpDown.Size = new Size(150, 27);
            delayNumericUpDown.TabIndex = 3;
            delayNumericUpDown.Value = new decimal(new int[] { 100, 0, 0, 0 });
            delayNumericUpDown.ValueChanged += delayNumericUpDown_ValueChanged;
            // 
            // panel1
            // 
            panel1.Controls.Add(numPointsNumericUpDown);
            panel1.Controls.Add(generateButton);
            panel1.Controls.Add(clearButton);
            panel1.Controls.Add(delayLabel);
            panel1.Controls.Add(startButton);
            panel1.Controls.Add(delayNumericUpDown);
            panel1.Dock = DockStyle.Top;
            panel1.Location = new Point(0, 0);
            panel1.Name = "panel1";
            panel1.Size = new Size(862, 58);
            panel1.TabIndex = 7;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(862, 518);
            Controls.Add(panel1);
            Controls.Add(canvasPictureBox);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            ((System.ComponentModel.ISupportInitialize)numPointsNumericUpDown).EndInit();
            ((System.ComponentModel.ISupportInitialize)canvasPictureBox).EndInit();
            ((System.ComponentModel.ISupportInitialize)delayNumericUpDown).EndInit();
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private NumericUpDown numPointsNumericUpDown;
        private Button generateButton;
        private Label delayLabel;
        private PictureBox canvasPictureBox;
        private Button startButton;
        private Button clearButton;
        private ContextMenuStrip contextMenuStrip1;
        private System.Windows.Forms.Timer timer;
        private NumericUpDown delayNumericUpDown;
        private Panel panel1;
    }
}
