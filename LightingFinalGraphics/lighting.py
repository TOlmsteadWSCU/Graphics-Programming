import sys
from array           import array
from ctypes          import c_void_p
from textwrap        import dedent
from OpenGL.GL       import *
from OpenGL.GLU      import *
from PyQt5.QtOpenGL  import QGLWidget
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui     import QMatrix4x4, QVector3D, QVector4D
import numpy as np
from pyassimp        import load



	
class Cow():
    def __init__(self):
        self.perspective = True
        self.directional = True
        self.count = 1
        self.model = QMatrix4x4()
        self.model.translate(5,5,0)
        self.model.scale(.5,.5,.5)
        self.initializeGL()
        

    def initializeModel(self):
        #model = loadObj('cow.obj', restart=self.restart)
        model = load('cow.obj').meshes[0]
        self.vertices = model.vertices
        self.faces = model.faces
        #print(self.vertices)
        

        # compute the normals of the faces
        # TODO: calculate smooth surface normals
        norms = []
        normsSmooth = []
        pointify = lambda n: [n.x(), n.y(), n.z()]
        for tri in range(len(self.faces)):
            # Here we're getting the indices for the current triangle
            ind0 = self.faces[tri][0]
            ind1 = self.faces[tri][1]
            ind2 = self.faces[tri][2]

            # Using those indices we can get the three points of the triangle
            p0 = QVector3D(self.vertices[ind0][0],
                           self.vertices[ind0][1],
                           self.vertices[ind0][2])

            p1 = QVector3D(self.vertices[ind1][0],
                           self.vertices[ind1][1],
                           self.vertices[ind1][2])

            p2 = QVector3D(self.vertices[ind2][0],
                           self.vertices[ind2][1],
                           self.vertices[ind2][2])

            p = p0 - p1
            q = p0 - p2
            n = pointify(QVector3D.normal(p, q))


            # Push the normal onto our normal array (once for each point)
            norms.append(n)
        self.norms = np.array(norms, dtype='float32')

        # create a new Vertex Array Object on the GPU which saves the attribute
        # layout of our vertices
        self.modelVao = glGenVertexArrays(1)
        glBindVertexArray(self.modelVao)

        # create a buffer on the GPU for position and color data
        vertexBuffer, indexBuffer, normalBuffer = glGenBuffers(3)

        # upload the data to the GPU, storing it in the buffer we just created
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.vertices.nbytes,
            self.vertices,
            GL_STATIC_DRAW
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.faces.nbytes,
            self.faces,
            GL_STATIC_DRAW
        )
        glBindBuffer(GL_ARRAY_BUFFER, normalBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.norms.nbytes,
            self.norms,
            GL_STATIC_DRAW
        )

        # load our vertex and fragment shaders into a program object on the GPU
        program = self.loadShaders()
        glUseProgram(program)
        self.modelProg = program

        # bind the attribute "position" (defined in our vertex shader) to the
        # currently bound buffer object, which contains our position data
        # this information is stored in our vertex array object
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
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

        # normal
        #glBindBuffer(GL_ARRAY_BUFFER, normalBuffer)
        '''
        normal = glGetAttribLocation(program, 'normal')
        glEnableVertexAttribArray(normal)
        glVertexAttribPointer(
            normal,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            c_void_p(0)
        )
        '''
        #glBindBuffer(GL_ARRAY_BUFFER, normalBuffer)
        #normal = glGetAttribLocation(program, 'normal')
        #glEnableVertexAttribArray(normal)
        #glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))

        # lighting variables
        # TODO: implement directional lighting
        #lightPos = glGetUniformLocation(program, 'lightPos')
        #glUniform3f(lightPos, 2, 1, 10)
        
        #modelColor = glGetUniformLocation(program, 'lightColor')
        #glUniform3f(modelColor, 1, 1, 1)  # white
        #modelAmbient = glGetUniformLocation(program, 'ambient')
        #glUniform1f(modelAmbient, 0.1)

        # reflection variables
        #diffuseConst = glGetUniformLocation(program, 'diffuse')
        #glUniform1f(diffuseConst, 0.55)
        # TODO: add a specular reflection coefficient
        '''
        specularConst = glGetUniformLocation(program, 'specular')
        glUniform1f(specularConst, 1)

        alphaCo = glGetUniformLocation(program, 'alpha')
        glUniform1f(alphaCo, 256)
        # TODO: add a uniform variable for the position of the camera
        self.cameraPosition = glGetUniformLocation(program, 'cameraPos')
        '''
        # project, model, and view transformation matrices
        self.modelProjMatLoc = glGetUniformLocation(program, "projection")
        self.modelModelMatLoc = glGetUniformLocation(program, "model")
        #self.directionalLoc = glGetUniformLocation(program, "b")

        #self.ambientLoc = glGetUniformLocation(program, "s")
        #self.diffuseLoc = glGetUniformLocation(program, "s2")
        #self.specLoc = glGetUniformLocation(program, "s3")
        
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        self.initializeModel()
        
    def loadShaders(self):
		
        # create a GL Program Object
        program = glCreateProgram()

        # vertex shader
        # TODO: implement directional lighting
        vs_source = dedent("""
            #version 330
            uniform mat4 projection;
            uniform mat4 model;
            in vec3 position;
            in vec3 normal;
            out vec3 pos;
            out vec3 norm;
            void main()
            {
               gl_Position = projection * model * vec4(position, 1.0);
               pos = (model * vec4(position, 1.0)).xyz;
               norm = normalize((transpose(inverse(model)) * vec4(normal, 0)).xyz);
            }\
        """)
        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, vs_source)
        glCompileShader(vs)
        glAttachShader(program, vs)
        if glGetShaderiv(vs, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vs))

        # fragment shader
        # TODO: compute specular lighting using the camera position
        # TODO: implement thje Phong reflection model
        fs_source = dedent("""
            #version 330
            
            //uniform vec3 lightPos;
            //uniform vec3 lightDir;
            //uniform vec3 lightColor;
            //uniform float ambient;
            //uniform float diffuse;
            //uniform float specular;
            //uniform float b;
            //uniform float s;
            //uniform float s2;
            //uniform float s3;
            //uniform vec3 cameraPos;
            //uniform float alpha;
            //in vec3 pos;
            //in vec3 norm;
            out vec4 color_out;
            void main()
            {
               //vec3 L = normalize(lightPos - (pos * b));
               //vec3 R = reflect(L, norm);
               //vec3 V = normalize(pos - cameraPos);
               //float diff = diffuse * clamp(dot(norm, L), 0, 1);
               //float spec = pow(specular * clamp(dot(R, V), 0, 1), alpha);
               //color_out = vec4(((ambient * s) + (diff * s2) + (spec * s3)) * lightColor, 1.0);
               color_out = vec4(1,1,1,1);

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


    def renderModel(self):
        glUseProgram(self.modelProg)
        glBindVertexArray(self.modelVao)
        #glDrawArrays(GL_TRIANGLES, 0, len(self.vertices))
        glDrawElements(
            GL_TRIANGLES,
            sum(map(len, [f for f in self.faces])),
            GL_UNSIGNED_INT,
            c_void_p(0)
        )


    def resizeGL(self, camera, cameraPos):
        
        b = 0 if self.directional else 1
        s = 1 if self.count == 1 else 0
        s2 = 1 if self.count == 2 else 0
        s3 = 1 if self.count == 3 else 0

        if self.count == 4:
            s = 1
            s2 = 1
            s3 = 1
        elif self.count == 5:
            self.count = 0
        
        
        cameraPos = QVector3D(2, 2, 2)
        camera.lookAt(cameraPos, QVector3D(0, 0, 0), QVector3D(0, 0, 1))
        # TODO: load the new camera position into the shaders
        
        glUseProgram(self.modelProg)
        glUniformMatrix4fv(
            self.modelProjMatLoc,
            1,
            GL_FALSE,
            array('f', camera.data()).tostring()
        )

        # model matrix
        glUseProgram(self.modelProg)
        glUniformMatrix4fv(
            self.modelModelMatLoc,
            1,
            GL_FALSE,
            array('f', self.model.data()).tostring()
        )

        '''
        glUseProgram(self.modelProg)
        glUniform1f(
            self.ambientLoc,
            s
        )

        glUseProgram(self.modelProg)
        glUniform1f(
            self.diffuseLoc,
            s2
        )

        glUseProgram(self.modelProg)
        glUniform1f(
            self.specLoc,
            s3
        )

        glUseProgram(self.modelProg)
        glUniform1f(
            self.directionalLoc,
            b
        )

        glUseProgram(self.modelProg)
        glUniform3f(
            self.cameraPosition,
            cameraPos.x(),
            cameraPos.y(),
            cameraPos.z()
        )
        '''
	
    def sizeof(self, a):
        return a.itemsize * len(a)
class Cube():
    def __init__(self, x, y, z, texture, w, h):
        self.restart = 0xFFFFFFFF
        self.model = QMatrix4x4()
        self.w = w
        self.h = h
        self.textureC = texture
        self.model.translate(x, y, z)
        self.initializeGL()
        
    def initializeCube(self):
        # compute the normals of the faces
        self.vertices = array('f', [
          # +y face
           0.5, 0.5,  0.5,  # 0
           0.5, 0.5, -0.5,  # 1
          -0.5, 0.5, -0.5,  # 2
          -0.5, 0.5,  0.5,  # 3
          # -y face
           0.5, -0.5,  0.5,  # 4
          -0.5, -0.5,  0.5,  # 5
          -0.5, -0.5, -0.5,  # 6
           0.5, -0.5, -0.5,  # 7
          # top
           0.5,  0.5, 0.5,  # 8
          -0.5,  0.5, 0.5,  # 9
          -0.5, -0.5, 0.5,  # 10
           0.5, -0.5, 0.5,  # 11
          # bottom
          -0.5, -0.5, -0.5,  # 12
          -0.5,  0.5, -0.5,  # 13
           0.5,  0.5, -0.5,  # 14
           0.5, -0.5, -0.5,  # 15
          # +x face
          0.5, -0.5,  0.5,  # 16
          0.5, -0.5, -0.5,  # 17
          0.5,  0.5, -0.5,  # 18
          0.5,  0.5,  0.5,  # 19
          # -x face
          -0.5, -0.5,  0.5,  # 20
          -0.5,  0.5,  0.5,  # 21
          -0.5,  0.5, -0.5,  # 22
          -0.5, -0.5, -0.5   # 23
        ])
        
        #self.norms = array('f', [
			# +y face
		#	[0,1,0],
			# -y face
		#	[0,-1,0],
			# +x face
		#	[1,0,0],
			# -x face
		#	[-1,0,0],
			# +z face
		#	[0,0,1],
			# -z face
		#	[0,0,-1]
		#])
            
        self.colors = array('f', [
          # top
          0,1,0,
          0,1,0,
          0,1,0,
          0,1,0,
          # bottom
          0,.5,0,
          0,.5,0,
          0,.5,0,
          0,.5,0,
          # front
          0,0,1,
          0,0,1,
          0,0,1,
          0,0,1,
          # back
          0,0,.5,
          0,0,.5,
          0,0,.5,
          0,0,.5,
          # right
          1,0,0,
          1,0,0,
          1,0,0,
          1,0,0,
          # left
          .5,0,0,
          .5,0,0,
          .5,0,0,
          .5,0,0
        ])
        self.indices = array('I', [
           0,  1,  2,  3, self.restart,
           4,  5,  6,  7, self.restart,
           8,  9, 10, 11, self.restart,
          12, 13, 14, 15, self.restart,
          16, 17, 18, 19, self.restart,
          20, 21, 22, 23
        ])

        # TODO: create an array of uv coordinates
        self.uvCoord = array('B', [
            # +y face
            0,0,
            0,1,
            1,1,
            1,0,
            # -y face
            0,0,
            0,1,
            1,1,
            1,0,
            # top
            0,0,
            0,1,
            1,1,
            1,0,
            # bottom
            0,0,
            0,1,
            1,1,
            1,0,
            # +x face
            0,0,
            0,1,
            1,1,
            1,0,
            # -x face
            0,0,
            0,1,
            1,1,
            1,0,
        ])
        

        # TODO: add texture data here

        # create a new Vertex Array Object on the GPU which saves the attribute
        # layout of our vertices
        self.cubeVao = glGenVertexArrays(1)
        glBindVertexArray(self.cubeVao)

        # create a buffer on the GPU for position and color data
        vertexBuffer, colorBuffer, indexBuffer, uvBuffer, normBuffer = glGenBuffers(5)

        # upload the data to the GPU, storing it in the buffer we just created
        # TODO: upload to uv data to a buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.sizeof(self.vertices),
            self.vertices.tostring(),
            GL_STATIC_DRAW
        )
        '''glBindBuffer(GL_ARRAY_BUFFER, colorBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.sizeof(self.colors),
            self.colors.tostring(),
            GL_STATIC_DRAW
        )'''
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.sizeof(self.indices),
            self.indices.tostring(),
            GL_STATIC_DRAW
        )
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.sizeof(self.uvCoord),
            self.uvCoord.tostring(),
            GL_STATIC_DRAW
        )
        # TODO: create a gl texture object
        self.texObject = glGenTextures(1);
        # TODO: bind to the texture object
        glBindTexture(GL_TEXTURE_2D, self.texObject);
        # TODO: load the texture data into the texture object
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            self.w,
            self.h,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            self.textureC,
        )
        # TODO: tell OpenGL how to sample from the texture
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        # load our vertex and fragment shaders into a program object on the GPU
        self.cubeProgram = self.loadShaders()
        glUseProgram(self.cubeProgram)

        # bind the attribute "position" (defined in our vertex shader) to the
        # currently bound buffer object, which contains our position data
        # this information is stored in our vertex array object
        # TODO: bind to the "uv" attribute defined in the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer)
        position = glGetAttribLocation(self.cubeProgram, 'position')
        glEnableVertexAttribArray(position)
        glVertexAttribPointer(
            position,
            3,
            GL_FLOAT,
            GL_FALSE,
            0,
            c_void_p(0)
        )
        #glBindBuffer(GL_ARRAY_BUFFER, colorBuffer)
        #color = glGetAttribLocation(self.cubeProgram, 'color')
        #glEnableVertexAttribArray(color)
        #glVertexAttribPointer(
         #   color,
          #  2,
           # GL_BYTE,
            #GL_FALSE,
            #0,
            #c_void_p(0)
       #)
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffer)
        uv = glGetAttribLocation(self.cubeProgram, 'uv')
        glEnableVertexAttribArray(uv)
        glVertexAttribPointer(
            uv,
            2,
            GL_BYTE,
            GL_FALSE,
            0,
            c_void_p(0)
       )
        # project, model, and view transformation matrices
        self.cubeProjection = glGetUniformLocation(self.cubeProgram, "projection")
        self.cubeTranslation = glGetUniformLocation(self.cubeProgram, "translation")


    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        self.initializeCube()

    def renderCube(self):
        glUseProgram(self.cubeProgram)
        glBindVertexArray(self.cubeVao)
        glDrawElements(
            GL_TRIANGLE_FAN,
            len(self.indices),
            GL_UNSIGNED_INT,
            c_void_p(0)
        )
        
    def loadShaders(self):
        # create a GL Program Object
        program = glCreateProgram()

        # vertex shader
        # TODO: add a vec2 input uv
        # TODO: pass uv to a vec2 output fragUV
        vs_source = dedent("""
            #version 330
            uniform mat4 projection;
            uniform mat4 translation;
            in vec3 position;
            in vec2 uv;
            out vec2 fragUV;
            void main()
            {
               gl_Position = projection * (translation * vec4(position, 1.0));
               fragUV = uv;
            }\
        """)
        vs = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vs, vs_source)
        glCompileShader(vs)
        glAttachShader(program, vs)
        if glGetShaderiv(vs, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(vs))

        # fragment shader
        # TODO: add a vec2 input fragUV
        # TODO: add a uniform sampler2D variable for the texture
        fs_source = dedent('''
            #version 330
            
            in vec2 fragUV;
            out vec4 color_out;
            uniform sampler2D tex;
            void main()
            {
               //color_out = vec4(fragColor, 1.0);
               //color_out = vec4(fragUV.x, fragUV.y, 0, 1);
               color_out = texture(tex, fragUV);
            }\
        ''')
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

    def sizeof(self, a):
        return a.itemsize * len(a)
    def resize(self, camera):
        glUseProgram(self.cubeProgram)
        glUniformMatrix4fv(
            self.cubeProjection,
            1,
            GL_FALSE,
            array('f', camera.data()).tostring()
        )
        glUseProgram(self.cubeProgram)
        glUniformMatrix4fv(
            self.cubeTranslation,
            1,
            GL_FALSE,
            array('f', self.model.data()).tostring()
        )


class Texture(QGLWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restart = 0xFFFFFFFF
    
    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
		
        camera = QMatrix4x4()
        camera.perspective(60, 4.0/3.0, 0.01, 100)
        cameraPos = QVector3D(10, 10, 10)
        camera.lookAt(cameraPos, QVector3D(0, 0, 0), QVector3D(0, 0, 1))

        self.cube1.resize(camera)
        self.cube2.resize(camera)
        self.cube3.resize(camera)
        self.cow1.resizeGL(camera, cameraPos)
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.cube1.renderCube()
        self.cube2.renderCube()
        self.cube3.renderCube()
        self.cow1.renderModel()

    
    def initializeGL(self):
		
        glEnable(GL_DEPTH_TEST)
        glPrimitiveRestartIndex(self.restart)
        glEnable(GL_PRIMITIVE_RESTART)       
        textureC = bytes  ([
            #row 1
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 2
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 3
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0xff, 0x75, 0x1a, 
            0xff, 0xff, 0xff, 
            0x00, 0x00, 0x00, 
            0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 4
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0x00, 0x00, 0x00, 
            0x00, 0x00, 0x00,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 5
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x00, 0x00, 0xff, 
            0xff, 0x00, 0x00, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 6
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 7
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 8
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 9
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0xff, 0x75, 0x1a,
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0x75, 0x1a, 
            0xb3, 0xb3, 0xb3, 
            0xff, 0xff, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 10
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff,
            0xff, 0x75, 0x1a,
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3,
            0xff, 0xff, 0xff, 
            0xFF, 0x75, 0x1a,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xff, 0xff, 0xff,
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3, 
            0x66, 0xb3, 0xff,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 11
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0xff, 0xff, 0xff, 
            0xb3, 0xb3, 0xb3, 
            0xb3, 0xb3, 0xb3,  
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a,  
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a,
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff,
            #row 12
            0x66, 0xb3, 0xff, 
            0x66, 0xb3, 0xff, 
            0xb3, 0xb3, 0xb3, 
            0xFF, 0x00, 0xFF,
            0xff, 0xff, 0xff, 
            0xb3, 0xb3, 0xb3, 
            0xff, 0x75, 0x1a, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0x66, 0xb3, 0xff,  
            0x66, 0xb3, 0xff, 
            #row 13
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a,  
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,  
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,  
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc,
            #row 14
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a, 
            0xff, 0x75, 0x1a,  
            0xff, 0x75, 0x1a,  
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc,
            #row 15
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xff, 0xff, 
            0xff, 0xff, 0xff, 
            0xff, 0x75, 0x1a,   
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3,
            0xb3, 0xb3, 0xb3,
            0xff, 0x75, 0x1a, 
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            #row 16
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0xe6, 0xcc,
            0xff, 0x75, 0x1a, 
            0xb3, 0xb3, 0xb3, 
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xff, 0xff,
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
            0xff, 0xe6, 0xcc, 
        ])
        self.cube1 = Cube(-3,0,0, textureC, 16, 16)
        self.cube2 = Cube(1,-2,0, textureC, 16, 16)
        self.cube3 = Cube(0,1,-1, textureC, 16, 16)
        self.cow1 = Cow()
        
    
    def keyPressEvent(self, event):
			 # TODO: implement key events here
        key = event.text()
        if key == 'p':
             self.cow1.perspective = not self.cow1.perspective
        elif key == 'x':
             self.cow1.model.rotate(5, 1, 0, 0)
        elif key == 'y':
             self.cow1.model.rotate(5, 0, 1, 0)
        elif key == 'z':
             self.cow1.model.rotate(5, 0, 0, 1)
        elif key == 'l':
             self.cow1.directional = not self.cow1.directional
        elif key == 'r':
             self.cow1.count += 1
			
        self.resizeGL(self.width(), self.height())
        self.update()	
	
	
if __name__ == '__main__':

    width = 640
    height = 480

    app = QApplication(sys.argv)
    w = Texture()
    w.show()

    sys.exit(app.exec_())
