//Devraj Mehta
//Sugarscape
//Serial version of CUDA ABM application to Sugarscape
//gcc -o ss -lm -Wall -framework GLUT -framework OpenGL ss.c

//version 1: implements growth and decay of sugar levels
//	mod: all models flat
//	mod: float capacity readded
//	mod: OpenGL added
//version 2: agents move around world
//	mod: movement is random
//	mod: metabolism added
//	mod: movement optimal within vision

//standard imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//OpenGL imports
#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>

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

//variable intializations
float *sugar_levels, *sugar_maximums; //flattened matrices
float fps; //frames per second: epoque
char wtitle[256]; //title of glut window
Agent *agent_list; //list of agents in world
int *agent_matrix; //map of agent positions
int n_left; //the number of agents left
long long int step; //counter for the number of iterations
int freeze_flag; //flag to stop updating
int num_steps;
int W, H, N;

//variables used in gl
double xmin,ymin,xmax,ymax;
int w,h; //screen size

float gauss(int x, int y, int x0, int y0, int sx, int sy)
{
	return exp(-0.5*(x-x0)*(x-x0)/sx/sx)*exp(-0.5*(y-y0)*(y-y0)/sy/sy);
}
void grow_sugar(float *s_levels, float *s_maximums)
{
	int i; //temp iter var
	float growth_rate = 0.1;
	for(i=0; i<W*H; i++) {
		s_levels[i] += growth_rate;
		if(s_levels[i] > s_maximums[i])
			s_levels[i] = s_maximums[i];
		/*		if(s_levels[i] > 0.5 && s_maximums[i] < 0.7) //making a ridge
		 s_levels[i] = 0.5;
		 if(s_levels[i] > 0.7 && s_maximums[i] < 0.9)
		 s_levels[i] = 0.7;
		 */	}
}
void update_agents(Agent *a_list, int *a_mat, float *s_levels)
{
	//declare variables
	int r, k, i, j, v, x, y; //temp iter var
	float b; //best sugar level seen
	int *bi; //chosen location of best sugar level seen
	int nb; //the number of best sugar levels seen
	float p; //amount of sugar agent can eat
	
	//allocate method memory for locations of best sugar levels
	bi = (int*)malloc((2*10)*(2*10)*sizeof(int));
	
	//loop over agents
	for(k=0; k<N; k++) {
		if(a_list[k].sugar > 0.0) { //agent is alive
			v = a_list[k].vision; x = a_list[k].x; y = a_list[k].y;
			//search for locations of best sugar
			b = s_levels[W * a_list[k].x + a_list[k].y]; //best known sugar level
			bi[0] = W * a_list[k].x + a_list[k].y;
			nb = 1;
			for(i = -v; i <= v /*&& b <= s*/; i++) {
				if(i+x >=0 && i+x < W) {
					for(j = -v; j <= v /*&& b <= s*/; j++) {
						if(j+y < H && j+y >= 0 && a_mat[W*(i+x)+j+y] == -1) { //check valid & vacant
							if(s_levels[W*(i+x)+j+y] > b) {
								b = s_levels[W*(i+x)+j+y];
								bi[0] = W*(i+x) + j+y;
								nb = 1;
							}
							else if(s_levels[W*(i+x)+j+y]==b) {
								bi[nb] = W*(i+x)+j+y;
								nb++;
							}
						}
					}
				}
			}
			
			//decide on location and move to it
			v = 1; //flag to continue looping over locations
			while(v) {
				r = random()%nb; //index in bi of index of best sugar
				if(a_mat[bi[r]] == k)
					v = 0;//simply don't move
				else if(a_mat[bi[r]] == -1) {
					a_mat[W * a_list[k].x + a_list[k].y] = -1;
					a_mat[bi[r]] = k;
					a_list[k].x = bi[r] / W;
					a_list[k].y = bi[r] % W;
					v = 0;
				}
			}
			
			//eat
			a_list[k].sugar -= a_list[k].metabolism;
			if(a_list[k].sugar <= 0.0) {
				a_mat[W * a_list[k].x + a_list[k].y] = -1;
				n_left--;
			}
			else {
				p = 1.0 - a_list[k].sugar;
				if(p > s_levels[W * a_list[k].x + a_list[k].y]) {
					a_list[k].sugar += s_levels[W * a_list[k].x + a_list[k].y];
					s_levels[W * a_list[k].x + a_list[k].y] = 0.0;
				}
				else {
					s_levels[W * a_list[k].x + a_list[k].y] -= p;
					a_list[k].sugar = 1.0;
				}
			}
		}
	}
}
void display(void)
{
	if(step==num_steps) {
		//exit program and release memory
		free(sugar_levels);
		free(sugar_maximums);
		free(agent_list);
		free(agent_matrix);
		exit(0);
	}
	
	//declare variables
	int x,y,z;
	double a,b,c,d;
	
	//begin time
	fps = (float)clock()/CLOCKS_PER_SEC;
	
	//display world
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_POINTS);
	for(x=0; x<w; x++) {
		for(y=0; y<h; y++) {
			z = W * (W * x / w) + H * y /h; //index translated for sugarscape
			a = sugar_levels[z];
			d = 0.0;
			if(agent_matrix[z]!=-1)
				d = agent_list[agent_matrix[z]].sugar;
			glColor3f(0.0, a, d);
			b = xmax * x / w - xmin; //coordinates translated for gl
			c = ymax * y / h - ymin;
			glVertex2f(b,c);
		}
	}
	
	//end rendering and display updated buffer contents
	glEnd();
	glutSwapBuffers();
	
	//check for permission
	if(freeze_flag)
		return;
	
	//run instructions
	grow_sugar(sugar_levels, sugar_maximums);
	update_agents(agent_list, agent_matrix, sugar_levels);
	
	//finish and display epoque
	fps = 1.0f / ((float)clock()/CLOCKS_PER_SEC - fps);
	printf("%f\n", 1.0f / fps);
	sprintf(wtitle, "Sugarscape (CPU) %d x %d  %d agents  %3.1f fps  step #%lld",W,H,n_left,fps,step);
	glutSetWindowTitle(wtitle);
	
	//increment counter
	step++;
}
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
	n_left = N;
}
void reset_sugar()
{
	int k; //temp iter var
	for(k=0;k<W*H;k++)
		sugar_levels[k] = 0;
}
void keyfunc(unsigned char key,int xscr,int yscr)
{
	if(key=='q')
	{
		//exit program and release memory
		free(sugar_levels);
		free(sugar_maximums);
		free(agent_list);
		free(agent_matrix);
		printf("\nq pressed; program exiting.\n");
		exit(0);
	}
	else if(key=='r')
	{
		//reset all sugar levels to zero and randomize agents
		reset_sugar();
		reset_agents();
		step = 0;
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
void mouse(int button,int state,int xscr,int yscr)
{
	int j, k; //temp iter vars
 	if(button==GLUT_LEFT_BUTTON)
 		if(state==GLUT_DOWN)
		{
			//set clicked upon sugar level to zero
			for(j=W*xscr/w-R; j<W*xscr/w+R; j++)
				if(j<W && j>=0)
					for(k=H*(h-yscr)/h-R; k<=H*(h-yscr)/h+R; k++)
						if(k<H && k>=0)
							sugar_levels[W * j + k] = 0;
			//glutPostRedisplay(); // callback
		}
	if(button==GLUT_RIGHT_BUTTON)
		if(state==GLUT_DOWN)
		{
			//print this cell's properties
			printf("cell location clicked %d, %d\n", W*xscr/w, H*(h-yscr)/h);
			printf("sugar at cell is %f of max %f\n", sugar_levels[W * (W*xscr/w) + (H*(h-yscr)/h)],
				   sugar_maximums[W * (W*xscr/w) + (H*(h-yscr)/h)]);
			if(agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]==-1)
				printf("no agent at cell.\n");
			else {
				printf("agent at cell has vision %d, metabolism %f, and current sugar level %f.\n",
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].vision,
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].metabolism,
					   agent_list[agent_matrix[W * (W*xscr/w) + (H*(h-yscr)/h)]].sugar);
			}
		}
}
void move(int xscr, int yscr)
{
	int j, k; //temp iter vars
	//set clicked upon sugar level to zero
	for(j=W*xscr/w-R; j<W*xscr/w+R; j++)
		if(j<W && j>=0)
			for(k=H*(h-yscr)/h-R; k<=H*(h-yscr)/h+R; k++)
				if(k<H && k>=0)
					sugar_levels[W * j + k] = 0;
}
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
	int i, j; //temp iter vars
	step = 0;
	freeze_flag = 0;
	
	//allocate matrices
	sugar_levels = (float*)malloc(		W*H*sizeof(float));
	sugar_maximums = (float*)malloc(	W*H*sizeof(float));
	agent_list = (Agent*)malloc(		N*sizeof(Agent));
	agent_matrix = (int*)malloc(		W*H*sizeof(int));
	
	//initialize matrices
	for(i=0;i<W;i++)
		for(j=0;j<H;j++)
			sugar_maximums[W*i+j] = gauss(i,j,W/4,H*3/4,W/5,H/5) + gauss(i,j,W*3/4,H/4,W/5,H/5);
	reset_sugar();
	reset_agents();
	
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
