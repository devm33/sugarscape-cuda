//Devraj Mehta
//Sugarscape
//Applies CUDA to ABM in Sugarscape

//Using pinned memory

//standard imports
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//OpenGL imports
#include <GL/glew.h>
#include <GLUT/glut.h> //for Mac

//preprocessor definitions
#define R 50 //radius of mouse agent

//agent model
typedef struct {
	int x;
	int y;
	float sugar;
	float metabolism;
	int vision;
} Agent;

//variable declarations
int sugar_g, agent_g; //the number of blocks needed for sugar and agent kernel
int sugar_b, agent_b; //number of threads per block
float *sugar_levels, *sugar_maximums; //flattened matrices on host
float *sl_d, *sm_d; //matrices copied to device
float fps; //frames per second: epoque
char wtitle[256]; //title of glut window
Agent *agent_list, *a_d; //list of agents in world on host and device
int *agent_matrix, *am_d; //matrix of agent locations
int freeze_flag; //flag to halt all updating
long long int step; //counter for the number of iterations
int num_steps;
int W; //latteral resolution of world
int H; //vertical resolution of world
int N; //number of agents

//variables used in gl
double xmin,ymin,xmax,ymax;
int w, h; //screen size

//gaussian function to determine layout of sugar
float gauss(int x, int y, int x0, int y0, int sx, int sy)
{
	return expf(-0.5*(x-x0)*(x-x0)/sx/sx)*expf(-0.5*(y-y0)*(y-y0)/sy/sy);
}

//kernel to grow sugar patches at each time step
__global__ void grow_sugar(float *s_levels, float *s_maximums)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; //index of sugar cell
	float growth_rate = 0.1f;
	s_levels[i] += growth_rate;
	if(s_levels[i] > s_maximums[i])
		s_levels[i] = s_maximums[i];
/*	if(s_levels[i] > 0.5 && s_maximums[i] < 0.7) //making a ridge
		s_levels[i] = 0.5;
	if(s_levels[i] > 0.7 && s_maximums[i] < 0.9)
		s_levels[i] = 0.7;*/
}

//kernel to updatae the agents' sugar levels
__global__ void feed_agents(Agent *a_list, int *a_mat, float *s_levels,
	int width, int height)
{
//declare variables
	int k; //agent's index
	float p; //amount of sugar agent can eat

//set index
	k = blockIdx.x * blockDim.x + threadIdx.x;

//if the agent is alive (this is bad for cuda simd)
	if(a_list[k].sugar > 0.0) {

//increment metabolism
		a_list[k].sugar -= a_list[k].metabolism;
//check if agent survived
		if(a_list[k].sugar <= 0.0) {
			a_mat[width * a_list[k].x + a_list[k].y] = -1;
		}
		else {
//if stil alive take sugar from current patch
			p = 1.0 - a_list[k].sugar;
			if(p > s_levels[width * a_list[k].x + a_list[k].y]) {
				a_list[k].sugar += s_levels[width * a_list[k].x + a_list[k].y];
				s_levels[width * a_list[k].x + a_list[k].y] = 0.0;
			}
			else {
				s_levels[width * a_list[k].x + a_list[k].y] -= p;
				a_list[k].sugar = 1.0;
			}
		}
	}
}

//kernel to update the agents' location
__global__ void move_agents(Agent *a_list, int *a_mat, float *s_levels,
		int width, int height)
{

//declare variables
	int i, j, k, x, y, f; //k: index of agent
	float b; //best sugar level seen
	int bx, by; //chosen location of best sugar level seen
	int v; //agent's vision
	k = blockIdx.x * blockDim.x + threadIdx.x;

//if the agent is alive (this is kinda bad for cuda simt)
	if(a_list[k].sugar > 0.0) {
		f=1;
		x = a_list[k].x;
		y = a_list[k].y;
		v = a_list[k].vision;
		while(f) {
			f=0;
			b = s_levels[width*x+y]; //best known sugar level
			bx = x; by = y;
			for(i = -v; i <= v /*&& b <= s*/; i++) {
				if(i+x >=0 && i+x < width) {
					for(j = -v; j <= v /*&& b <= s*/; j++) {
						if(j+y < height && j+y >= 0 && a_mat[width*(i+x)+j+y] == -1) { //check valid & vacant
							if(s_levels[width*(i+x)+j+y] > b) {
								b = s_levels[width*(i+x)+j+y];
								bx = i+x;
								by = j+y;
							}
						}
					}
				}
			}

			//move to location
			if(a_mat[width*bx+by] == k); //simply dont move
			else if(atomicExch(a_mat+width*bx+by, k) == -1) { //atomic test and set operation
				a_mat[width * a_list[k].x + a_list[k].y] = -1;
				a_list[k].x = bx;
				a_list[k].y = by;
			}
			else
				f=1;
		}
	}
}

//method to display world in opengl
void display(void)
{
	if(step==num_steps) {
//exit program and release memory
		cudaFreeHost(sugar_levels);
		free(sugar_maximums);
		cudaFreeHost(agent_list);
		cudaFreeHost(agent_matrix);
		cudaFree(sl_d);
		cudaFree(sm_d);
		cudaFree(a_d);
		cudaFree(am_d);
		exit(0);
		}

//check to see if permitted
	if(freeze_flag)
		return;

//declare variables
	int x,y,z;
	long long int n_left=0;
	double a,b,c,d;
	cudaError_t cet;

//begin time
	fps = (float)clock()/CLOCKS_PER_SEC;

//run instructions on device
    grow_sugar<<< sugar_g, sugar_b>>>(sl_d, sm_d); //asynchronus, nonblocking

//block until all threads finish
	cudaThreadSynchronize();
	
//run instructions on device
    feed_agents<<< agent_g, agent_b>>>(a_d, am_d, sl_d, W, H); //asynchronus, nonblocking

//block until all threads finish
	cudaThreadSynchronize();
		
//run instructions on device
    move_agents<<< agent_g, agent_b>>>(a_d, am_d, sl_d, W, H); //asynchronus, nonblocking

//block until all threads finish
	cudaThreadSynchronize();

//copy updated matrices from device to host
	cudaMemcpy(sugar_levels,	sl_d,	W*H*sizeof(float),	cudaMemcpyDeviceToHost);
	cudaMemcpy(agent_matrix,	am_d,	W*H*sizeof(int),	cudaMemcpyDeviceToHost);
	//cudaMemcpy(agent_list,		a_d,	N*sizeof(Agent),	cudaMemcpyDeviceToHost);

//display world
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);
	for(x=0; x<w; x++) {
		for(y=0; y<h; y++) {
			z = W * (W * x / w) + H * y /h; //coordinates translated for sugarscape
			a = sugar_levels[z];
			d = 0.0;
			if(agent_matrix[z] != -1) {
				d = 1.0;
				n_left++;
			}
			glColor3f(0.0, a, d);
			b = xmax * x / w - xmin; //coordinates translated for gl
			c = ymax * y / h - ymin;
			glVertex2f(b,c);
		}
	}
	glEnd();
	
//scale counter
	if(W<w && H<h)
		n_left = (W*H)*n_left/(w*h);
	else if(W>w || H>h)
		n_left = N;

//finish and display epoque
	fps = 1.0 / ((float)clock()/CLOCKS_PER_SEC - fps);
	printf("%f\n", 1.0f/ fps);
	sprintf(wtitle, "Sugarscape (GPU) %d x %d  %lld agents  %3.1f fps  step #%lld",W,H,n_left,fps, step);
	glutSetWindowTitle(wtitle);

//end rendering and display updated buffer contents
	glutSwapBuffers();

//increment counter
	step++;

//check for errors in cuda
	cet = cudaGetLastError();
	if(cet != cudaSuccess)
		printf("CUDA ERROR: %s\n", cudaGetErrorString(cet));
}

//sets all agents to initial states
void reset_agents()
{
	int j, k, m, n; //temp iter var
	for(k=0; k<W*H; k++)
		agent_matrix[k] = -1;
	for(k=0; k<N; k++) {
		j=1;
		while(j) {
			m = random() % H;
			n = random() % W;
			if(agent_matrix[ W * m + n] == -1)
				j=0;
		}
		agent_matrix[W * m + n] = k;
		agent_list[k].x = m;
		agent_list[k].y = n;
		agent_list[k].sugar = 1.0; 
		agent_list[k].metabolism = 0.001 * (random() % 900) + 0.1;
		agent_list[k].vision = random() % 9 + 1;
	}
	cudaMemcpy(a_d, agent_list,			N*sizeof(Agent),	cudaMemcpyHostToDevice);
	cudaMemcpy(am_d, agent_matrix,		W*H*sizeof(int),	cudaMemcpyHostToDevice);
}

//zeros all sugar levels
void reset_sugar()
{
	int k; //temp iter var
	for(k=0;k<W*H;k++)
		sugar_levels[k] = 0;
	cudaMemcpy(sl_d, sugar_levels, W*H*sizeof(float),	cudaMemcpyHostToDevice);
}

//method to register opengl key events
void keyfunc(unsigned char key,int xscr,int yscr)
{
	if(key=='q')
	{
//exit program and release memory
		cudaFreeHost(sugar_levels);
		free(sugar_maximums);
		cudaFreeHost(agent_list);
		cudaFreeHost(agent_matrix);
		cudaFree(sl_d);
		cudaFree(sm_d);
		cudaFree(a_d);
		cudaFree(am_d);
		printf("\nq pressed; program exiting.\n");
		exit(0);
	}
	else if(key=='r')
	{
//reset all sugar levels to zero
		reset_agents();
		reset_sugar();
	}
	else if(key=='s')
	{
//reset all sugar levels to zero
		reset_sugar();
	}
	else if(key=='a')
	{
//randomize and reset all agents
		reset_agents();
	}
	else if(key=='p')
	{
//(un)freeze all updating
		freeze_flag = freeze_flag ? 0 : 1;
	}
}

//method to register opengl mouse events
void mouse(int button,int state,int xscr,int yscr)
{
	int j, k; //temp iter vars
 	if(button==GLUT_LEFT_BUTTON)
	{
 		if(state==GLUT_DOWN)
		{
//set clicked upon sugar level to zero
			for(j=W*xscr/w-R; j<W*xscr/w+R; j++)
				if(j<W && j>=0)
					for(k=H*(h-yscr)/h-R; k<=H*(h-yscr)/h+R; k++)
						if(k<H && k>=0)
							sugar_levels[W * j + k] = 0;
			cudaMemcpy(sl_d, sugar_levels, W*H*sizeof(float), cudaMemcpyHostToDevice);
			//glutPostRedisplay(); // callback
		}
	}
	else if(button==GLUT_RIGHT_BUTTON)
	{
		if(state==GLUT_DOWN)
		{
//print this cell's properties
			printf("(%d, %d)\n", W*xscr/w, H*(h-yscr)/h);
			printf("\tpatch:\n\t\tsugar\t%f\n\t\tmax\t%f\n", sugar_levels[W * (W*xscr/w) + (H*(h-yscr)/h)],
				   sugar_maximums[W * (W*xscr/w) + (H*(h-yscr)/h)]);
			if(agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]==-1)
				printf("\tagent:\n\t\tnone\n");
			else {
				printf("\tagent:\n\t\tvision\t%d\n\t\tmetab\t%f\n\t\tsugar\t%f\n",
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].vision,
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].metabolism,
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].sugar);
			}
			printf("\tmatrix:\n\t\tindex\t%d\n\t\tvalue\t%d\n", W * (W*xscr/w) + (H*(h-yscr)/h), agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]);
		}
	}
}

//method to register opengl mouse movement events
void move(int xscr, int yscr)
{
	int j, k; //temp iter vars
//set clicked upon sugar level to zero
	for(j=W*xscr/w-R; j<W*xscr/w+R; j++)
		if(j<W && j>=0)
			for(k=H*(h-yscr)/h-R; k<=H*(h-yscr)/h+R; k++)
				if(k<H && k>=0)
					sugar_levels[W * j + k] = 0;
	cudaMemcpy(sl_d, sugar_levels, W*H*sizeof(float), cudaMemcpyHostToDevice);
}

//method to handle the screen being resized
void reshape(int wscr,int hscr)
{
	w=wscr; h=hscr;
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	
	xmin=ymin=0.0; xmax=ymax=1.0;
	if(w<=h)
		ymax=1.0*(GLfloat)h/(GLfloat)w;
	else
		xmax=1.0*(GLfloat)w/(GLfloat)h;
	
	gluOrtho2D(xmin,xmax,ymin,ymax);
	glMatrixMode(GL_MODELVIEW);
}

//main method
int main(int argc, char* argv[])
{
//fetching for input
	if(argc!=3)
	{
		printf("please input N then number of steps\n");
		return 0;
	}
	N = atoi(argv[1]);
	W = N; H = N;
	w = N; h = N;
	N = N*N;
	num_steps = atoi(argv[2]);

//declare variables
	int i, j, k, l; //temp iter vars
	cudaDeviceProp dp; //properties for device
	int max_threads; //the maximum number of threads per block

//set best device (the one with the most multiprocessors)
	cudaGetDeviceCount(&i);
	k=0; l=0;
	for(j=0; j<i; j++) {
		cudaGetDeviceProperties(&dp, j);
		if(dp.multiProcessorCount > l) {
			l = dp.multiProcessorCount;
			k = j;
		}
	}
	cudaSetDevice(k);
	cudaGetDeviceProperties(&dp, k);
	printf("Operating on %s\n", dp.name);

//define variables
	max_threads = dp.maxThreadsPerBlock;
//find the smallest x so that x*y=N, y<M, & x & y are both integers: perhaps there is a better way
	sugar_b = max_threads;
	while((W*H) % sugar_b != 0)
		sugar_b--;	
	sugar_g = W*H/sugar_b;
	agent_b = max_threads;
	while(N % agent_b != 0)
		agent_b--;
	agent_g = N/agent_b;
	freeze_flag = 0;
	step = 0;

//allocate matrices
	cudaMallocHost((void**)&sugar_levels, W*H*sizeof(float));
	cudaMalloc((void**)&sl_d,		W*H*sizeof(float));

	sugar_maximums = (float*)malloc(	W*H*sizeof(float));
	cudaMalloc((void**)&sm_d,		W*H*sizeof(float));

	cudaMallocHost((void**)&agent_list, N*sizeof(Agent));
	cudaMalloc((void**)&a_d,		N*sizeof(Agent));

	cudaMallocHost((void**)&agent_matrix, W*H*sizeof(int));
	cudaMalloc((void**)&am_d,		W*H*sizeof(int));

//initialize matrices on host
	memset(sugar_levels, 0, W*H*sizeof(float));
	for(i=0;i<W;i++)
		for(j=0;j<H;j++)
			sugar_maximums[W*i+j] = gauss(i,j,W/4,H*3/4,W/5,H/5) + gauss(i,j,W*3/4,H/4,W/5,H/5);
	reset_sugar();
	reset_agents();
	
//copy matrices to device
	cudaMemcpy(sm_d, sugar_maximums,	W*H*sizeof(float), cudaMemcpyHostToDevice);
	
//setup OpenGL
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(w,h);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Sugarscape");
	glClearColor(1.0,1.0,1.0,0.0);
	
//gl callback functions
   	glutDisplayFunc(display);
	glutIdleFunc(display);
  	glutMouseFunc(mouse);
	glutMotionFunc(move);
 	glutKeyboardFunc(keyfunc);
	glutReshapeFunc(reshape);

//begin looping sugarscape
	glutMainLoop();
	
	return 0;
}
