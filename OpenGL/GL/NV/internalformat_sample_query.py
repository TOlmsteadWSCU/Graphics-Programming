'''OpenGL extension NV.internalformat_sample_query

This module customises the behaviour of the 
OpenGL.raw.GL.NV.internalformat_sample_query to provide a more 
Python-friendly API

The official definition of this extension is available here:
http://www.opengl.org/registry/specs/NV/internalformat_sample_query.txt
'''
from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.internalformat_sample_query import *
from OpenGL.raw.GL.NV.internalformat_sample_query import _EXTENSION_NAME

def glInitInternalformatSampleQueryNV():
    '''Return boolean indicating whether this extension is available'''
    from OpenGL import extensions
    return extensions.hasGLExtension( _EXTENSION_NAME )

# INPUT glGetInternalformatSampleivNV.params size not checked against bufSize
glGetInternalformatSampleivNV=wrapper.wrapper(glGetInternalformatSampleivNV).setInputArraySize(
    'params', None
)
### END AUTOGENERATED SECTION