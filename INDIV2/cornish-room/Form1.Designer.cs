namespace cornish_room;

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
    /// Required method for Designer support - do not modify
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent()
    {
        pictureBoxScene = new System.Windows.Forms.PictureBox();
        groupBox1 = new System.Windows.Forms.GroupBox();
        radioButtonCube1Nothing = new System.Windows.Forms.RadioButton();
        radioButtonCube1Transparency = new System.Windows.Forms.RadioButton();
        radioButtonCube1Specularity = new System.Windows.Forms.RadioButton();
        groupBox2 = new System.Windows.Forms.GroupBox();
        radioButtonCube2Nothing = new System.Windows.Forms.RadioButton();
        radioButtonCube2Transparency = new System.Windows.Forms.RadioButton();
        radioButtonCube2Specularity = new System.Windows.Forms.RadioButton();
        panel1 = new System.Windows.Forms.Panel();
        groupBox11 = new System.Windows.Forms.GroupBox();
        checkBoxSecondLight = new System.Windows.Forms.CheckBox();
        label3 = new System.Windows.Forms.Label();
        numericUpDownX = new System.Windows.Forms.NumericUpDown();
        label2 = new System.Windows.Forms.Label();
        numericUpDownY = new System.Windows.Forms.NumericUpDown();
        label1 = new System.Windows.Forms.Label();
        numericUpDownZ = new System.Windows.Forms.NumericUpDown();
        groupBox9 = new System.Windows.Forms.GroupBox();
        radioButtonRightWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonRightWallSpecularity = new System.Windows.Forms.RadioButton();
        radioButtonRightWallTransparency = new System.Windows.Forms.RadioButton();
        groupBox10 = new System.Windows.Forms.GroupBox();
        radioButtonLeftWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonlLeftWallTransparency = new System.Windows.Forms.RadioButton();
        radioButtonLeftWallSpecularity = new System.Windows.Forms.RadioButton();
        groupBox7 = new System.Windows.Forms.GroupBox();
        radioButtonUpWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonUpWallSpecularity = new System.Windows.Forms.RadioButton();
        radioButtonUpWallTransparency = new System.Windows.Forms.RadioButton();
        groupBox8 = new System.Windows.Forms.GroupBox();
        radioButtonDownWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonDownWallTransparency = new System.Windows.Forms.RadioButton();
        radioButtonDownWallSpecularity = new System.Windows.Forms.RadioButton();
        groupBox5 = new System.Windows.Forms.GroupBox();
        radioButtonFrontWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonFrontWallSpecularity = new System.Windows.Forms.RadioButton();
        radioButtonFrontWallTransparency = new System.Windows.Forms.RadioButton();
        buttonStartRender = new System.Windows.Forms.Button();
        groupBox6 = new System.Windows.Forms.GroupBox();
        radioButtonBackWallNothing = new System.Windows.Forms.RadioButton();
        radioButtonBackWallTransparency = new System.Windows.Forms.RadioButton();
        radioButtonBackWallSpecularity = new System.Windows.Forms.RadioButton();
        groupBox3 = new System.Windows.Forms.GroupBox();
        radioButtonSphere1Nothing = new System.Windows.Forms.RadioButton();
        radioButtonSphere1Specularity = new System.Windows.Forms.RadioButton();
        radioButtonSphere1Transparency = new System.Windows.Forms.RadioButton();
        groupBox4 = new System.Windows.Forms.GroupBox();
        radioButtonSphere2Nothing = new System.Windows.Forms.RadioButton();
        radioButtonSphere2Transparency = new System.Windows.Forms.RadioButton();
        radioButtonSphere2Specularity = new System.Windows.Forms.RadioButton();
        ((System.ComponentModel.ISupportInitialize)pictureBoxScene).BeginInit();
        groupBox1.SuspendLayout();
        groupBox2.SuspendLayout();
        panel1.SuspendLayout();
        groupBox11.SuspendLayout();
        ((System.ComponentModel.ISupportInitialize)numericUpDownX).BeginInit();
        ((System.ComponentModel.ISupportInitialize)numericUpDownY).BeginInit();
        ((System.ComponentModel.ISupportInitialize)numericUpDownZ).BeginInit();
        groupBox9.SuspendLayout();
        groupBox10.SuspendLayout();
        groupBox7.SuspendLayout();
        groupBox8.SuspendLayout();
        groupBox5.SuspendLayout();
        groupBox6.SuspendLayout();
        groupBox3.SuspendLayout();
        groupBox4.SuspendLayout();
        SuspendLayout();
        // 
        // pictureBoxScene
        // 
        pictureBoxScene.BackColor = System.Drawing.SystemColors.Control;
        pictureBoxScene.Location = new System.Drawing.Point(304, 0);
        pictureBoxScene.Name = "pictureBoxScene";
        pictureBoxScene.Size = new System.Drawing.Size(960, 680);
        pictureBoxScene.TabIndex = 0;
        pictureBoxScene.TabStop = false;
        // 
        // groupBox1
        // 
        groupBox1.Controls.Add(radioButtonCube1Nothing);
        groupBox1.Controls.Add(radioButtonCube1Transparency);
        groupBox1.Controls.Add(radioButtonCube1Specularity);
        groupBox1.Location = new System.Drawing.Point(10, 10);
        groupBox1.Name = "groupBox1";
        groupBox1.Size = new System.Drawing.Size(140, 110);
        groupBox1.TabIndex = 1;
        groupBox1.TabStop = false;
        groupBox1.Text = "Куб";
        // 
        // radioButtonCube1Nothing
        // 
        radioButtonCube1Nothing.Location = new System.Drawing.Point(9, 19);
        radioButtonCube1Nothing.Name = "radioButtonCube1Nothing";
        radioButtonCube1Nothing.Size = new System.Drawing.Size(104, 24);
        radioButtonCube1Nothing.TabIndex = 2;
        radioButtonCube1Nothing.Text = "Ничего";
        radioButtonCube1Nothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonCube1Transparency
        // 
        radioButtonCube1Transparency.Checked = true;
        radioButtonCube1Transparency.Location = new System.Drawing.Point(9, 79);
        radioButtonCube1Transparency.Name = "radioButtonCube1Transparency";
        radioButtonCube1Transparency.Size = new System.Drawing.Size(104, 24);
        radioButtonCube1Transparency.TabIndex = 1;
        radioButtonCube1Transparency.TabStop = true;
        radioButtonCube1Transparency.Text = "Прозрачность";
        radioButtonCube1Transparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonCube1Specularity
        // 
        radioButtonCube1Specularity.Location = new System.Drawing.Point(9, 49);
        radioButtonCube1Specularity.Name = "radioButtonCube1Specularity";
        radioButtonCube1Specularity.Size = new System.Drawing.Size(104, 24);
        radioButtonCube1Specularity.TabIndex = 0;
        radioButtonCube1Specularity.Text = "Зеркальность\r\n";
        radioButtonCube1Specularity.UseVisualStyleBackColor = true;
        // 
        // groupBox2
        // 
        groupBox2.Controls.Add(radioButtonCube2Nothing);
        groupBox2.Controls.Add(radioButtonCube2Transparency);
        groupBox2.Controls.Add(radioButtonCube2Specularity);
        groupBox2.Location = new System.Drawing.Point(156, 10);
        groupBox2.Name = "groupBox2";
        groupBox2.Size = new System.Drawing.Size(140, 110);
        groupBox2.TabIndex = 2;
        groupBox2.TabStop = false;
        groupBox2.Text = "Куб поменьше";
        // 
        // radioButtonCube2Nothing
        // 
        radioButtonCube2Nothing.Location = new System.Drawing.Point(10, 19);
        radioButtonCube2Nothing.Name = "radioButtonCube2Nothing";
        radioButtonCube2Nothing.Size = new System.Drawing.Size(104, 24);
        radioButtonCube2Nothing.TabIndex = 5;
        radioButtonCube2Nothing.Text = "Ничего";
        radioButtonCube2Nothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonCube2Transparency
        // 
        radioButtonCube2Transparency.Checked = true;
        radioButtonCube2Transparency.Location = new System.Drawing.Point(10, 79);
        radioButtonCube2Transparency.Name = "radioButtonCube2Transparency";
        radioButtonCube2Transparency.Size = new System.Drawing.Size(104, 24);
        radioButtonCube2Transparency.TabIndex = 4;
        radioButtonCube2Transparency.TabStop = true;
        radioButtonCube2Transparency.Text = "Прозрачность";
        radioButtonCube2Transparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonCube2Specularity
        // 
        radioButtonCube2Specularity.Location = new System.Drawing.Point(10, 49);
        radioButtonCube2Specularity.Name = "radioButtonCube2Specularity";
        radioButtonCube2Specularity.Size = new System.Drawing.Size(104, 24);
        radioButtonCube2Specularity.TabIndex = 3;
        radioButtonCube2Specularity.Text = "Зеркальность\r\n";
        radioButtonCube2Specularity.UseVisualStyleBackColor = true;
        // 
        // panel1
        // 
        panel1.Controls.Add(groupBox11);
        panel1.Controls.Add(groupBox9);
        panel1.Controls.Add(groupBox10);
        panel1.Controls.Add(groupBox7);
        panel1.Controls.Add(groupBox8);
        panel1.Controls.Add(groupBox5);
        panel1.Controls.Add(buttonStartRender);
        panel1.Controls.Add(groupBox6);
        panel1.Controls.Add(groupBox3);
        panel1.Controls.Add(groupBox4);
        panel1.Controls.Add(groupBox1);
        panel1.Controls.Add(groupBox2);
        panel1.Controls.Add(pictureBoxScene);
        panel1.Dock = System.Windows.Forms.DockStyle.Fill;
        panel1.Location = new System.Drawing.Point(0, 0);
        panel1.Name = "panel1";
        panel1.Size = new System.Drawing.Size(1264, 681);
        panel1.TabIndex = 3;
        // 
        // groupBox11
        // 
        groupBox11.Controls.Add(checkBoxSecondLight);
        groupBox11.Controls.Add(label3);
        groupBox11.Controls.Add(numericUpDownX);
        groupBox11.Controls.Add(label2);
        groupBox11.Controls.Add(numericUpDownY);
        groupBox11.Controls.Add(label1);
        groupBox11.Controls.Add(numericUpDownZ);
        groupBox11.Location = new System.Drawing.Point(10, 590);
        groupBox11.Name = "groupBox11";
        groupBox11.Size = new System.Drawing.Size(286, 44);
        groupBox11.TabIndex = 12;
        groupBox11.TabStop = false;
        groupBox11.Text = "Свет";
        // 
        // checkBoxSecondLight
        // 
        checkBoxSecondLight.Location = new System.Drawing.Point(9, 15);
        checkBoxSecondLight.Name = "checkBoxSecondLight";
        checkBoxSecondLight.Size = new System.Drawing.Size(84, 23);
        checkBoxSecondLight.TabIndex = 6;
        checkBoxSecondLight.Text = "Второй свет";
        checkBoxSecondLight.TextAlign = System.Drawing.ContentAlignment.BottomLeft;
        checkBoxSecondLight.UseVisualStyleBackColor = true;
        // 
        // label3
        // 
        label3.Location = new System.Drawing.Point(99, 12);
        label3.Name = "label3";
        label3.Size = new System.Drawing.Size(13, 23);
        label3.TabIndex = 5;
        label3.Text = "X";
        label3.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
        // 
        // numericUpDownX
        // 
        numericUpDownX.Location = new System.Drawing.Point(118, 12);
        numericUpDownX.Maximum = new decimal(new int[] { 9, 0, 0, 0 });
        numericUpDownX.Minimum = new decimal(new int[] { 9, 0, 0, 0 });
        numericUpDownX.Name = "numericUpDownX";
        numericUpDownX.Size = new System.Drawing.Size(36, 23);
        numericUpDownX.TabIndex = 4;
        numericUpDownX.Value = new decimal(new int[] { 9, 0, 0, 0 });
        // 
        // label2
        // 
        label2.Location = new System.Drawing.Point(160, 12);
        label2.Name = "label2";
        label2.Size = new System.Drawing.Size(13, 23);
        label2.TabIndex = 3;
        label2.Text = "Y";
        label2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
        // 
        // numericUpDownY
        // 
        numericUpDownY.Location = new System.Drawing.Point(179, 12);
        numericUpDownY.Maximum = new decimal(new int[] { 9, 0, 0, 0 });
        numericUpDownY.Minimum = new decimal(new int[] { 9, 0, 0, -2147483648 });
        numericUpDownY.Name = "numericUpDownY";
        numericUpDownY.Size = new System.Drawing.Size(36, 23);
        numericUpDownY.TabIndex = 2;
        // 
        // label1
        // 
        label1.Location = new System.Drawing.Point(221, 12);
        label1.Name = "label1";
        label1.Size = new System.Drawing.Size(13, 23);
        label1.TabIndex = 1;
        label1.Text = "Z";
        label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
        // 
        // numericUpDownZ
        // 
        numericUpDownZ.Location = new System.Drawing.Point(240, 12);
        numericUpDownZ.Maximum = new decimal(new int[] { 9, 0, 0, 0 });
        numericUpDownZ.Minimum = new decimal(new int[] { 9, 0, 0, -2147483648 });
        numericUpDownZ.Name = "numericUpDownZ";
        numericUpDownZ.Size = new System.Drawing.Size(36, 23);
        numericUpDownZ.TabIndex = 0;
        // 
        // groupBox9
        // 
        groupBox9.Controls.Add(radioButtonRightWallNothing);
        groupBox9.Controls.Add(radioButtonRightWallSpecularity);
        groupBox9.Controls.Add(radioButtonRightWallTransparency);
        groupBox9.Location = new System.Drawing.Point(10, 474);
        groupBox9.Name = "groupBox9";
        groupBox9.Size = new System.Drawing.Size(140, 110);
        groupBox9.TabIndex = 10;
        groupBox9.TabStop = false;
        groupBox9.Text = "Правая стена";
        // 
        // radioButtonRightWallNothing
        // 
        radioButtonRightWallNothing.Checked = true;
        radioButtonRightWallNothing.Location = new System.Drawing.Point(9, 18);
        radioButtonRightWallNothing.Name = "radioButtonRightWallNothing";
        radioButtonRightWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonRightWallNothing.TabIndex = 5;
        radioButtonRightWallNothing.TabStop = true;
        radioButtonRightWallNothing.Text = "Ничего";
        radioButtonRightWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonRightWallSpecularity
        // 
        radioButtonRightWallSpecularity.Location = new System.Drawing.Point(9, 48);
        radioButtonRightWallSpecularity.Name = "radioButtonRightWallSpecularity";
        radioButtonRightWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonRightWallSpecularity.TabIndex = 3;
        radioButtonRightWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonRightWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // radioButtonRightWallTransparency
        // 
        radioButtonRightWallTransparency.Location = new System.Drawing.Point(9, 78);
        radioButtonRightWallTransparency.Name = "radioButtonRightWallTransparency";
        radioButtonRightWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonRightWallTransparency.TabIndex = 4;
        radioButtonRightWallTransparency.Text = "Прозрачность";
        radioButtonRightWallTransparency.UseVisualStyleBackColor = true;
        // 
        // groupBox10
        // 
        groupBox10.Controls.Add(radioButtonLeftWallNothing);
        groupBox10.Controls.Add(radioButtonlLeftWallTransparency);
        groupBox10.Controls.Add(radioButtonLeftWallSpecularity);
        groupBox10.Location = new System.Drawing.Point(156, 474);
        groupBox10.Name = "groupBox10";
        groupBox10.Size = new System.Drawing.Size(140, 110);
        groupBox10.TabIndex = 11;
        groupBox10.TabStop = false;
        groupBox10.Text = "Левая стена";
        // 
        // radioButtonLeftWallNothing
        // 
        radioButtonLeftWallNothing.Checked = true;
        radioButtonLeftWallNothing.Location = new System.Drawing.Point(10, 18);
        radioButtonLeftWallNothing.Name = "radioButtonLeftWallNothing";
        radioButtonLeftWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonLeftWallNothing.TabIndex = 8;
        radioButtonLeftWallNothing.TabStop = true;
        radioButtonLeftWallNothing.Text = "Ничего";
        radioButtonLeftWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonlLeftWallTransparency
        // 
        radioButtonlLeftWallTransparency.Location = new System.Drawing.Point(10, 78);
        radioButtonlLeftWallTransparency.Name = "radioButtonlLeftWallTransparency";
        radioButtonlLeftWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonlLeftWallTransparency.TabIndex = 7;
        radioButtonlLeftWallTransparency.Text = "Прозрачность";
        radioButtonlLeftWallTransparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonLeftWallSpecularity
        // 
        radioButtonLeftWallSpecularity.Location = new System.Drawing.Point(10, 48);
        radioButtonLeftWallSpecularity.Name = "radioButtonLeftWallSpecularity";
        radioButtonLeftWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonLeftWallSpecularity.TabIndex = 6;
        radioButtonLeftWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonLeftWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // groupBox7
        // 
        groupBox7.Controls.Add(radioButtonUpWallNothing);
        groupBox7.Controls.Add(radioButtonUpWallSpecularity);
        groupBox7.Controls.Add(radioButtonUpWallTransparency);
        groupBox7.Location = new System.Drawing.Point(10, 358);
        groupBox7.Name = "groupBox7";
        groupBox7.Size = new System.Drawing.Size(140, 110);
        groupBox7.TabIndex = 8;
        groupBox7.TabStop = false;
        groupBox7.Text = "Верхняя стена";
        // 
        // radioButtonUpWallNothing
        // 
        radioButtonUpWallNothing.Checked = true;
        radioButtonUpWallNothing.Location = new System.Drawing.Point(9, 18);
        radioButtonUpWallNothing.Name = "radioButtonUpWallNothing";
        radioButtonUpWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonUpWallNothing.TabIndex = 5;
        radioButtonUpWallNothing.TabStop = true;
        radioButtonUpWallNothing.Text = "Ничего";
        radioButtonUpWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonUpWallSpecularity
        // 
        radioButtonUpWallSpecularity.Location = new System.Drawing.Point(9, 48);
        radioButtonUpWallSpecularity.Name = "radioButtonUpWallSpecularity";
        radioButtonUpWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonUpWallSpecularity.TabIndex = 3;
        radioButtonUpWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonUpWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // radioButtonUpWallTransparency
        // 
        radioButtonUpWallTransparency.Location = new System.Drawing.Point(9, 78);
        radioButtonUpWallTransparency.Name = "radioButtonUpWallTransparency";
        radioButtonUpWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonUpWallTransparency.TabIndex = 4;
        radioButtonUpWallTransparency.Text = "Прозрачность";
        radioButtonUpWallTransparency.UseVisualStyleBackColor = true;
        // 
        // groupBox8
        // 
        groupBox8.Controls.Add(radioButtonDownWallNothing);
        groupBox8.Controls.Add(radioButtonDownWallTransparency);
        groupBox8.Controls.Add(radioButtonDownWallSpecularity);
        groupBox8.Location = new System.Drawing.Point(156, 358);
        groupBox8.Name = "groupBox8";
        groupBox8.Size = new System.Drawing.Size(140, 110);
        groupBox8.TabIndex = 9;
        groupBox8.TabStop = false;
        groupBox8.Text = "Нижняя стена";
        // 
        // radioButtonDownWallNothing
        // 
        radioButtonDownWallNothing.Checked = true;
        radioButtonDownWallNothing.Location = new System.Drawing.Point(10, 18);
        radioButtonDownWallNothing.Name = "radioButtonDownWallNothing";
        radioButtonDownWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonDownWallNothing.TabIndex = 8;
        radioButtonDownWallNothing.TabStop = true;
        radioButtonDownWallNothing.Text = "Ничего";
        radioButtonDownWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonDownWallTransparency
        // 
        radioButtonDownWallTransparency.Location = new System.Drawing.Point(10, 78);
        radioButtonDownWallTransparency.Name = "radioButtonDownWallTransparency";
        radioButtonDownWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonDownWallTransparency.TabIndex = 7;
        radioButtonDownWallTransparency.Text = "Прозрачность";
        radioButtonDownWallTransparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonDownWallSpecularity
        // 
        radioButtonDownWallSpecularity.Location = new System.Drawing.Point(10, 48);
        radioButtonDownWallSpecularity.Name = "radioButtonDownWallSpecularity";
        radioButtonDownWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonDownWallSpecularity.TabIndex = 6;
        radioButtonDownWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonDownWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // groupBox5
        // 
        groupBox5.Controls.Add(radioButtonFrontWallNothing);
        groupBox5.Controls.Add(radioButtonFrontWallSpecularity);
        groupBox5.Controls.Add(radioButtonFrontWallTransparency);
        groupBox5.Location = new System.Drawing.Point(10, 242);
        groupBox5.Name = "groupBox5";
        groupBox5.Size = new System.Drawing.Size(140, 110);
        groupBox5.TabIndex = 6;
        groupBox5.TabStop = false;
        groupBox5.Text = "Передняя стена";
        // 
        // radioButtonFrontWallNothing
        // 
        radioButtonFrontWallNothing.Location = new System.Drawing.Point(9, 18);
        radioButtonFrontWallNothing.Name = "radioButtonFrontWallNothing";
        radioButtonFrontWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonFrontWallNothing.TabIndex = 5;
        radioButtonFrontWallNothing.Text = "Ничего";
        radioButtonFrontWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonFrontWallSpecularity
        // 
        radioButtonFrontWallSpecularity.Checked = true;
        radioButtonFrontWallSpecularity.Location = new System.Drawing.Point(9, 48);
        radioButtonFrontWallSpecularity.Name = "radioButtonFrontWallSpecularity";
        radioButtonFrontWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonFrontWallSpecularity.TabIndex = 3;
        radioButtonFrontWallSpecularity.TabStop = true;
        radioButtonFrontWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonFrontWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // radioButtonFrontWallTransparency
        // 
        radioButtonFrontWallTransparency.Location = new System.Drawing.Point(9, 78);
        radioButtonFrontWallTransparency.Name = "radioButtonFrontWallTransparency";
        radioButtonFrontWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonFrontWallTransparency.TabIndex = 4;
        radioButtonFrontWallTransparency.Text = "Прозрачность";
        radioButtonFrontWallTransparency.UseVisualStyleBackColor = true;
        // 
        // buttonStartRender
        // 
        buttonStartRender.Location = new System.Drawing.Point(10, 640);
        buttonStartRender.Name = "buttonStartRender";
        buttonStartRender.Size = new System.Drawing.Size(288, 30);
        buttonStartRender.TabIndex = 5;
        buttonStartRender.Text = "Запустить рендер сцены";
        buttonStartRender.UseVisualStyleBackColor = true;
        buttonStartRender.Click += buttonStartRender_Click;
        // 
        // groupBox6
        // 
        groupBox6.Controls.Add(radioButtonBackWallNothing);
        groupBox6.Controls.Add(radioButtonBackWallTransparency);
        groupBox6.Controls.Add(radioButtonBackWallSpecularity);
        groupBox6.Location = new System.Drawing.Point(156, 242);
        groupBox6.Name = "groupBox6";
        groupBox6.Size = new System.Drawing.Size(140, 110);
        groupBox6.TabIndex = 7;
        groupBox6.TabStop = false;
        groupBox6.Text = "Задняя стена";
        // 
        // radioButtonBackWallNothing
        // 
        radioButtonBackWallNothing.Checked = true;
        radioButtonBackWallNothing.Location = new System.Drawing.Point(10, 18);
        radioButtonBackWallNothing.Name = "radioButtonBackWallNothing";
        radioButtonBackWallNothing.Size = new System.Drawing.Size(104, 24);
        radioButtonBackWallNothing.TabIndex = 8;
        radioButtonBackWallNothing.TabStop = true;
        radioButtonBackWallNothing.Text = "Ничего";
        radioButtonBackWallNothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonBackWallTransparency
        // 
        radioButtonBackWallTransparency.Location = new System.Drawing.Point(10, 78);
        radioButtonBackWallTransparency.Name = "radioButtonBackWallTransparency";
        radioButtonBackWallTransparency.Size = new System.Drawing.Size(104, 24);
        radioButtonBackWallTransparency.TabIndex = 7;
        radioButtonBackWallTransparency.Text = "Прозрачность";
        radioButtonBackWallTransparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonBackWallSpecularity
        // 
        radioButtonBackWallSpecularity.Location = new System.Drawing.Point(10, 48);
        radioButtonBackWallSpecularity.Name = "radioButtonBackWallSpecularity";
        radioButtonBackWallSpecularity.Size = new System.Drawing.Size(104, 24);
        radioButtonBackWallSpecularity.TabIndex = 6;
        radioButtonBackWallSpecularity.Text = "Зеркальность\r\n";
        radioButtonBackWallSpecularity.UseVisualStyleBackColor = true;
        // 
        // groupBox3
        // 
        groupBox3.Controls.Add(radioButtonSphere1Nothing);
        groupBox3.Controls.Add(radioButtonSphere1Specularity);
        groupBox3.Controls.Add(radioButtonSphere1Transparency);
        groupBox3.Location = new System.Drawing.Point(10, 126);
        groupBox3.Name = "groupBox3";
        groupBox3.Size = new System.Drawing.Size(140, 110);
        groupBox3.TabIndex = 3;
        groupBox3.TabStop = false;
        groupBox3.Text = "Шар";
        // 
        // radioButtonSphere1Nothing
        // 
        radioButtonSphere1Nothing.Location = new System.Drawing.Point(9, 18);
        radioButtonSphere1Nothing.Name = "radioButtonSphere1Nothing";
        radioButtonSphere1Nothing.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere1Nothing.TabIndex = 5;
        radioButtonSphere1Nothing.Text = "Ничего";
        radioButtonSphere1Nothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonSphere1Specularity
        // 
        radioButtonSphere1Specularity.Checked = true;
        radioButtonSphere1Specularity.Location = new System.Drawing.Point(9, 48);
        radioButtonSphere1Specularity.Name = "radioButtonSphere1Specularity";
        radioButtonSphere1Specularity.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere1Specularity.TabIndex = 3;
        radioButtonSphere1Specularity.TabStop = true;
        radioButtonSphere1Specularity.Text = "Зеркальность\r\n";
        radioButtonSphere1Specularity.UseVisualStyleBackColor = true;
        // 
        // radioButtonSphere1Transparency
        // 
        radioButtonSphere1Transparency.Location = new System.Drawing.Point(9, 78);
        radioButtonSphere1Transparency.Name = "radioButtonSphere1Transparency";
        radioButtonSphere1Transparency.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere1Transparency.TabIndex = 4;
        radioButtonSphere1Transparency.Text = "Прозрачность";
        radioButtonSphere1Transparency.UseVisualStyleBackColor = true;
        // 
        // groupBox4
        // 
        groupBox4.Controls.Add(radioButtonSphere2Nothing);
        groupBox4.Controls.Add(radioButtonSphere2Transparency);
        groupBox4.Controls.Add(radioButtonSphere2Specularity);
        groupBox4.Location = new System.Drawing.Point(156, 126);
        groupBox4.Name = "groupBox4";
        groupBox4.Size = new System.Drawing.Size(140, 110);
        groupBox4.TabIndex = 4;
        groupBox4.TabStop = false;
        groupBox4.Text = "Шар поменьше";
        // 
        // radioButtonSphere2Nothing
        // 
        radioButtonSphere2Nothing.Location = new System.Drawing.Point(10, 18);
        radioButtonSphere2Nothing.Name = "radioButtonSphere2Nothing";
        radioButtonSphere2Nothing.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere2Nothing.TabIndex = 8;
        radioButtonSphere2Nothing.Text = "Ничего";
        radioButtonSphere2Nothing.UseVisualStyleBackColor = true;
        // 
        // radioButtonSphere2Transparency
        // 
        radioButtonSphere2Transparency.Location = new System.Drawing.Point(10, 78);
        radioButtonSphere2Transparency.Name = "radioButtonSphere2Transparency";
        radioButtonSphere2Transparency.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere2Transparency.TabIndex = 7;
        radioButtonSphere2Transparency.Text = "Прозрачность";
        radioButtonSphere2Transparency.UseVisualStyleBackColor = true;
        // 
        // radioButtonSphere2Specularity
        // 
        radioButtonSphere2Specularity.Checked = true;
        radioButtonSphere2Specularity.Location = new System.Drawing.Point(10, 48);
        radioButtonSphere2Specularity.Name = "radioButtonSphere2Specularity";
        radioButtonSphere2Specularity.Size = new System.Drawing.Size(104, 24);
        radioButtonSphere2Specularity.TabIndex = 6;
        radioButtonSphere2Specularity.TabStop = true;
        radioButtonSphere2Specularity.Text = "Зеркальность\r\n";
        radioButtonSphere2Specularity.UseVisualStyleBackColor = true;
        // 
        // Form1
        // 
        AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
        AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
        AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
        ClientSize = new System.Drawing.Size(1264, 681);
        Controls.Add(panel1);
        MaximizeBox = false;
        MaximumSize = new System.Drawing.Size(1280, 720);
        MinimumSize = new System.Drawing.Size(1280, 720);
        Text = "Form1";
        ((System.ComponentModel.ISupportInitialize)pictureBoxScene).EndInit();
        groupBox1.ResumeLayout(false);
        groupBox2.ResumeLayout(false);
        panel1.ResumeLayout(false);
        groupBox11.ResumeLayout(false);
        ((System.ComponentModel.ISupportInitialize)numericUpDownX).EndInit();
        ((System.ComponentModel.ISupportInitialize)numericUpDownY).EndInit();
        ((System.ComponentModel.ISupportInitialize)numericUpDownZ).EndInit();
        groupBox9.ResumeLayout(false);
        groupBox10.ResumeLayout(false);
        groupBox7.ResumeLayout(false);
        groupBox8.ResumeLayout(false);
        groupBox5.ResumeLayout(false);
        groupBox6.ResumeLayout(false);
        groupBox3.ResumeLayout(false);
        groupBox4.ResumeLayout(false);
        ResumeLayout(false);
    }

    private System.Windows.Forms.CheckBox checkBoxSecondLight;

    private System.Windows.Forms.Label label2;
    private System.Windows.Forms.NumericUpDown numericUpDownY;
    private System.Windows.Forms.Label label3;
    private System.Windows.Forms.NumericUpDown numericUpDownX;

    private System.Windows.Forms.Label label1;

    private System.Windows.Forms.GroupBox groupBox11;
    private System.Windows.Forms.NumericUpDown numericUpDownZ;

    private System.Windows.Forms.RadioButton radioButtonSphere1Nothing;
    private System.Windows.Forms.RadioButton radioButtonSphere1Transparency;
    private System.Windows.Forms.RadioButton radioButtonSphere1Specularity;
    private System.Windows.Forms.RadioButton radioButtonSphere2Nothing;
    private System.Windows.Forms.RadioButton radioButtonSphere2Transparency;
    private System.Windows.Forms.RadioButton radioButtonSphere2Specularity;
    private System.Windows.Forms.GroupBox groupBox5;
    private System.Windows.Forms.RadioButton radioButtonFrontWallNothing;
    private System.Windows.Forms.RadioButton radioButtonFrontWallSpecularity;
    private System.Windows.Forms.RadioButton radioButtonFrontWallTransparency;
    private System.Windows.Forms.GroupBox groupBox6;
    private System.Windows.Forms.RadioButton radioButtonBackWallNothing;
    private System.Windows.Forms.RadioButton radioButtonBackWallTransparency;
    private System.Windows.Forms.RadioButton radioButtonBackWallSpecularity;
    private System.Windows.Forms.GroupBox groupBox7;
    private System.Windows.Forms.RadioButton radioButtonUpWallNothing;
    private System.Windows.Forms.RadioButton radioButtonUpWallSpecularity;
    private System.Windows.Forms.RadioButton radioButtonUpWallTransparency;
    private System.Windows.Forms.GroupBox groupBox8;
    private System.Windows.Forms.RadioButton radioButtonDownWallNothing;
    private System.Windows.Forms.RadioButton radioButtonDownWallTransparency;
    private System.Windows.Forms.RadioButton radioButtonDownWallSpecularity;
    private System.Windows.Forms.GroupBox groupBox9;
    private System.Windows.Forms.RadioButton radioButtonRightWallNothing;
    private System.Windows.Forms.RadioButton radioButtonRightWallSpecularity;
    private System.Windows.Forms.RadioButton radioButtonRightWallTransparency;
    private System.Windows.Forms.GroupBox groupBox10;
    private System.Windows.Forms.RadioButton radioButtonLeftWallNothing;
    private System.Windows.Forms.RadioButton radioButtonlLeftWallTransparency;
    private System.Windows.Forms.RadioButton radioButtonLeftWallSpecularity;

    private System.Windows.Forms.RadioButton radioButtonCube1Nothing;
    private System.Windows.Forms.RadioButton radioButtonCube2Nothing;
    private System.Windows.Forms.RadioButton radioButtonCube2Transparency;
    private System.Windows.Forms.RadioButton radioButtonCube2Specularity;

    private System.Windows.Forms.RadioButton radioButtonCube1Transparency;

    private System.Windows.Forms.RadioButton radioButtonCube1Specularity;

    private System.Windows.Forms.GroupBox groupBox3;
    private System.Windows.Forms.GroupBox groupBox4;
    private System.Windows.Forms.Button buttonStartRender;

    private System.Windows.Forms.Panel panel1;

    private System.Windows.Forms.GroupBox groupBox1;
    private System.Windows.Forms.GroupBox groupBox2;

    private System.Windows.Forms.PictureBox pictureBoxScene;

    #endregion
}