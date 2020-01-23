#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <GL/glew.h>
#undef GLFW_DLL
#include <GLFW/glfw3.h>
#include <renderer.hpp>

#define TEX_W 1280
#define TEX_H 736
#define TEX_SIZE (TEX_W * TEX_H)

//Exit safely, closing GLFW nicely
void safeExit();
//Load and compile vertex and fragment shader
GLuint LoadShaders(const char * vertFilePath, const char * fragFilePath);

int main(int argc, char const *argv[]) {
    //Init GLFW
    if (!glfwInit()) {
        puts("GLFW3 failed to initialize!");
        exit(EXIT_FAILURE);
    }

    //Init GLFW window
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(TEX_W, TEX_H, "Raytracing", NULL, NULL);
    if (!window) {
        puts("Could not open GLFW window");
        safeExit();
    }
    glfwMakeContextCurrent(window);

	//Extension handler GLEW
	glewExperimental = GL_TRUE;
    glewInit();
    
    //Get and display version information
	const GLubyte* renderer = glGetString(GL_RENDERER); // get renderer string
	const GLubyte* version = glGetString(GL_VERSION); // version as a string
	printf("GLSL version supported: %s\n", (char *)glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Renderer: %s\n", renderer);
	printf("OpenGL version supported %s\n", version);
    printf("Max texture size: %d\n", GL_MAX_TEXTURE_SIZE);
    
    //Setup for OpenGL
	glEnable(GL_DEPTH_TEST); //Enable depth-testing
	glEnable(GL_TEXTURE_3D);
	glDepthFunc(GL_LESS); //Depth-testing interprets a smaller value as "closer"

	//Create screen quad
	static const GLfloat gVertices[] = {
		-1, -1, 0,
		-1,  1, 0,
		 1,  1, 0,
		 1, -1, 0
	};
	static const GLubyte gIndices[] = {
		0, 1, 2,
		0, 4, 3
    };
    
    //Compile shaders
    GLuint programID = LoadShaders(
		"../resource/shaders/default.vert", 
		"../resource/shaders/default.frag");
    glUseProgram(programID);
    
    //Texture Data
	const uint texbytes = TEX_SIZE * 3 * sizeof(GLfloat);
    GLfloat *texData = (GLfloat *)malloc(texbytes);

	//Render to texture
	Renderer render(TEX_W, TEX_H);
	render.render(texData);
    
	//Create texture
	GLuint tex;
	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEX_W, TEX_H, 0, GL_RGB, GL_FLOAT, texData);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

	//Handlers
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glUniform1i(glGetUniformLocation(programID, "textureSampler"), 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    //Render loop
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS
        && !glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Draw the view quad
        glVertexPointer(3, GL_FLOAT, 0, gVertices);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, gIndices);

        GLenum err = glGetError();
        if (err) {
            printf("OPENGL ERROR: %s\n", (char *)glewGetErrorString(err));
            break;
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //Exit
    safeExit();
}

void safeExit() {
	glfwTerminate();
	exit(EXIT_SUCCESS);
}

GLuint LoadShaders(const char * vertFilePath, const char * fragFilePath) {

	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read the Vertex Shader code from the file
	std::string VertexShaderCode;
	std::ifstream VertexShaderStream(vertFilePath, std::ios::in);
	if(VertexShaderStream.is_open()){
		std::stringstream sstr;
		sstr << VertexShaderStream.rdbuf();
		VertexShaderCode = sstr.str();
		VertexShaderStream.close();
	}else{
		printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertFilePath);
		getchar();
		return 0;
	}

	// Read the Fragment Shader code from the file
	std::string FragmentShaderCode;
	std::ifstream FragmentShaderStream(fragFilePath, std::ios::in);
	if(FragmentShaderStream.is_open()){
		std::stringstream sstr;
		sstr << FragmentShaderStream.rdbuf();
		FragmentShaderCode = sstr.str();
		FragmentShaderStream.close();
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	printf("Compiling shader : %s\n", vertFilePath);
	char const * VertexSourcePointer = VertexShaderCode.c_str();
	glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
	glCompileShader(VertexShaderID);

	// Check Vertex Shader
	glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		printf("%s\n", &VertexShaderErrorMessage[0]);
	}

	// Compile Fragment Shader
	printf("Compiling shader : %s\n", fragFilePath);
	char const * FragmentSourcePointer = FragmentShaderCode.c_str();
	glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
	glCompileShader(FragmentShaderID);

	// Check Fragment Shader
	glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
		printf("%s\n", &FragmentShaderErrorMessage[0]);
	}

	// Link the program
	printf("Linking program\n");
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);

	// Check the program
	glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if ( InfoLogLength > 0 ){
		std::vector<char> ProgramErrorMessage(InfoLogLength+1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		printf("%s\n", &ProgramErrorMessage[0]);
	}
	
	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);

	return ProgramID;
}