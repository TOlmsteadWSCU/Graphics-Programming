import sys
from array import array
from ctypes import c_void_p
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5.QtGui import QImage, qRgb
from PyQt5.QtOpenGL import QGLWidget
from PyQt5.QtWidgets import QApplication
from textwrap import dedent


# create a function that draws a triangle via software rasterization
def softwareRasterization(width, height, vertices, colors):
    image = QImage(width, height, QImage.Format_RGB32)
    # TODO: compute the bounding box around the given vertices
    yMin = 1000
    yMax = 0
    xMin = 1000
    xMax = 0

    for i in range(0, len(vertices), 3):
        if vertices[i] > xMax:
            xMax = vertices[i]
        if vertices[i] < xMin:
            xMin = vertices[i]

    for i in range(1, len(vertices), 3):
        if vertices[i] > yMax:
            yMax = vertices[i]
        if vertices[i] < yMin:
            yMin = vertices[i]

    # TODO: compute the barycentric coordinates for each point in the box
    x1 = vertices[0]
    x2 = vertices[3]
    x3 = vertices[6]
    y1 = vertices[1]
    y2 = vertices[4]
    y3 = vertices[7]
    # c = qRgb(255, 0, 0)
    # c2 = qRgb(0, 255, 0)
    # c3 = qRgb(0, 0, 255)

    for y in range(yMin, yMax + 1):
        for x in range(xMin, xMax + 1):

            lam1 = (((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)))
            lam2 = (((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)))
            lam3 = 1 - lam1 - lam2
            c = ((colors[0] * 255) * lam1) + ((colors[3] * 255) * lam2) + ((colors[6] * 255) * lam3)
            c2 = ((colors[1] * 255) * lam1) + ((colors[4] * 255) * lam2) + ((colors[7] * 255) * lam3)
            c3 = ((colors[2] * 255) * lam1) + ((colors[5] * 255) * lam2) + ((colors[8] * 255) * lam3)
            cf = qRgb(c, c2, c3)

            cf = qRgb(c, c2, c3)

            if lam1 > 0 and lam2 > 0 and lam3 > 0:
                image.setPixel(x, y, cf)

    # TODO: color the points that are inside of the triangle
    if image.save("triangle.jpg", None, 100):
        print("Output triangle.jpg")
    else:
        print("Unable to save triangle.jpg")


# extend QGLWidget to draw a triangle via hardware rasterization
class HardwareRasterizationWidget(QGLWidget):
    def __init__(self, vertices, colors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vertices = array('f')
        # TODO: convert the input coordinate to normalized device coordinates
        self.vertices.append(-1 + vertices[0] * (2 / width))
        self.vertices.append(1 - vertices[1] * (2 / height))
        self.vertices.append(0)
        self.vertices.append(-1 + vertices[3] * (2 / width))
        self.vertices.append(1 - vertices[4] * (2 / height))
        self.vertices.append(0)
        self.vertices.append(-1 + vertices[6] * (2 / width))
        self.vertices.append(1 - vertices[7] * (2 / height))
        self.vertices.append(0)

        self.colors = array('f', colors)

    def _sizeof(self, a):
        return a.itemsize * len(a)

    def initializeGL(self):

        verticesSize = self._sizeof(self.vertices)
        colorSize = self._sizeof(self.colors)

        # create a new Vertex Array Object on the GPU which saves the attribute
        # layout of our vertices
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # create a buffer on the GPU for position and color data
        dataBuffer = glGenBuffers(1)

        # upload the data to the GPU, storing it in the buffer we just created
        # TODO: upload the color data into the GPU buffer as well
        glBindBuffer(GL_ARRAY_BUFFER, dataBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            verticesSize + colorSize,
            None,
            GL_STATIC_DRAW
        )
        glBufferSubData(
            GL_ARRAY_BUFFER,
            0,
            verticesSize,
            self.vertices.tostring()
        )
        glBufferSubData(
            GL_ARRAY_BUFFER,
            verticesSize,
            colorSize,
            self.colors.tostring()
        )

        # load our vertex and fragment shaders into a program object on the GPU
        program = self.loadShaders()

        # bind the attribute "position" (defined in our vertex shader) to the
        # currently bound buffer object, which contains our position data
        # this information is stored in our vertex array object
        position = glGetAttribLocation(program, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(
            position,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            c_void_p(0)
        )

        # TODO: bind the attribute "color" to the buffer object
        color = glGetAttribLocation(program, 'color')
        glEnableVertexAttribArray(color)
        glVertexAttribPointer(
            color,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            c_void_p(verticesSize)
        )

    def loadShaders(self):
        # create a GL Program Object
        program = glCreateProgram()

        # vertex shader
        # TODO: add a color input and color output
        vs_source = dedent("""
            #version 330
            in vec3 position;
            in vec3 color;
            out vec3 newColor;
            void main()
            {
               gl_Position = vec4(position, 1.0);
               newColor = color;
            }\

        """)

        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, vs_source)
        glCompileShader(vs)
        glAttachShader(program, vs)
        if glGetShaderiv(vs, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vs))

        # fragment shader
        # TODO: add a color input with the same name as the vertex output
        fs_source = dedent("""
            #version 330
            in vec3 newColor;
            void main()
            {
               gl_FragColor = vec4(newColor, 1.0);


            }\
        """)

        fs = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(fs, fs_source)
        glCompileShader(fs)
        glAttachShader(program, fs)
        if glGetShaderiv(fs, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(fs))

        # use the program
        glLinkProgram(program)
        glUseProgram(program)

        return program

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDrawArrays(GL_TRIANGLES, 0, 3)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)


if __name__ == "__main__":
    width = 640
    height = 480

    # TODO: prompt the user for 3 points and colors separated by spaces
    user = input("Input a points separated by spaces: ")
    userList = user.split(" ")

    user2 = input("Input a second point: ")
    userList2 = user2.split(" ")

    user3 = input("Input a final point: ")
    userList3 = user3.split(" ")

    userColor = input("Input a color for point 1: ")
    userColorList = userColor.split(" ")

    userColor2 = input("Input a color for point 2: ")
    userColorList2 = userColor2.split(" ")

    userColor3 = input("input a color for point 3: ")
    userColorList3 = userColor3.split(" ")

    userNum = [i for i in userList if i.isdigit()]
    userNum2 = [i for i in userList2 if i.isdigit()]
    userNum3 = [i for i in userList3 if i.isdigit()]

    userColorNum = [i for i in userColorList if i.isdigit()]
    userColorNum2 = [i for i in userColorList2 if i.isdigit()]
    userColorNum3 = [i for i in userColorList3 if i.isdigit()]

    # TODO: validate input and parse into the vertices and colors lists


    vertices = [
        int(userNum[0]), int(userNum[1]), 0,
        int(userNum2[0]), int(userNum2[1]), 0,
        int(userNum3[0]), int(userNum3[1]), 0
    ]

    print(vertices)
    ##    vertices = [
    ##        50, 50, 0,    # vertice 1
    ##        600, 20, 0,   # vertice 2
    ##        300, 400, 0   # vertice 3
    ##    ]
    ##    colors = [
    ##        1, 0, 0,  # color 1
    ##        0, 1, 0,  # color 2
    ##        0, 0, 1   # color 3
    ##    ]

    colors = [
        int(userColorNum[0]), int(userColorNum[1]), int(userColorNum[2]),
        int(userColorNum2[0]), int(userColorNum2[1]), int(userColorNum2[2]),
        int(userColorNum3[0]), int(userColorNum3[1]), int(userColorNum3[2])
    ]

    softwareRasterization(width, height, vertices, colors)

    app = QApplication(sys.argv)
    w = HardwareRasterizationWidget(vertices, colors)
    pRatio = w.devicePixelRatio()
    w.resize(width / pRatio, height / pRatio)
    w.show()

    sys.exit(app.exec_())
