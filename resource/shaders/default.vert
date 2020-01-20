#version 400

layout (location = 0) in vec3 VertexPosition;
   
out vec2 texCoord;

void main() {
    texCoord = VertexPosition.xy;
    gl_Position = vec4(VertexPosition, 1.0);
}