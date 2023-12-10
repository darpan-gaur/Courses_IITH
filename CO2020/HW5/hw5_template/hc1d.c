#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void grid(int nx, double xst, double xen, double *x, double *dx)
{
  int i;
  
  *dx = (xen-xst)/(double)(nx-1);

  for(i=0; i<nx; i++)
    x[i] = (double)i * (*dx); // ensure x[0] == 0.0 and x[nx-1] == 1.0
}

void enforce_bcs(int nx, double *x, double *T)
{
  T[0] = -1.0;
  T[nx-1] = 1.0;
}

void set_initial_condition(int nx, double *x, double *T)
{
  int i;

  for(i=0; i<nx; i++)
    T[i] = tanh((x[i]-0.5)/0.05);

  enforce_bcs(nx,x,T); //ensure BCs are satisfied at t = 0
}

void get_rhs(int nx, double dx, double *x, double *T, double *rhs)
{
  int i;
  double dxsq = dx*dx;

  // rhs at i=1
  // fourth order scheme
  rhs[1] = (10.0*T[0] - 15.0*T[1] - 4.0*T[2] + 14.0*T[3] - 6.0*T[4] + T[5])/(12.0*dxsq);

  // compute rhs. For this problem, d2T/dx2
  for(i=2; i<nx-2; i++){
    // rhs[i] = (T[i+1]+T[i-1]-2.0*T[i])/dxsq;
    // fourth order scheme
    rhs[i] = (-T[i+2]+16.0*T[i+1]-30.0*T[i]+16.0*T[i-1]-T[i-2])/(12.0*dxsq);
  }

  // rhs at i=nx-2
  // fourth order scheme
  rhs[nx-2] = (10.0*T[nx-1] - 15.0*T[nx-2] - 4.0*T[nx-3] + 14.0*T[nx-4] - 6.0*T[nx-5] + T[nx-6])/(12.0*dxsq);
}

void timestep_Euler(int nx, double dt, double dx, double *x, double *T, double *rhs)
{

  int i;

  // compute rhs
  get_rhs(nx,dx,x,T,rhs);

  // (Forward) Euler scheme
  for(i=0; i<nx; i++)
    T[i] = T[i] + dt*rhs[i];  // T^(it+1)[i] = T^(it)[i] + dt * rhs[i];

  // set Dirichlet BCs
  enforce_bcs(nx,x,T);

}

double get_l2err_norm(int nx, double *arr1, double *arr2)
{
    /*
    Function to compute the L2 error norm between two arrays
    */
    double l2err = 0.0, val;
    int i;

    for (i = 0; i < nx; i++)
      {
          val = arr1[i] - arr2[i];
          l2err += val * val;
      }
    // printf("l2err = %lf\n", l2err);
    l2err = l2err / ((double)(nx));
    l2err = sqrt(l2err);

    return l2err;
}

void exactSolution(int nx, double ten, double *x, double *Tex){
  /*
  Function to compute the exact solution
  */
  int i;
  
  for (i=0;i<nx;i++) {
    Tex[i] = erf((x[i]-0.5)/sqrt(4.0*ten));
  }
}

void output_soln(int nx, int it, double tcurr, double *x, double *T, double *Tex)
{
  int i;
  FILE* fp;
  char fname[100];

  sprintf(fname, "T_x_%04d.dat", it);
  //printf("\n%s\n", fname);

  fp = fopen(fname, "w");
  for(i=0; i<nx; i++)
    fprintf(fp, "%lf %lf %lf\n", x[i], T[i], Tex[i]);
  fclose(fp);

  printf("Done writing solution for time step = %d\n", it);
}

int main()
{

  int nx;
  double *x, *T, *rhs, tst, ten, xst, xen, dx, dt, tcurr;
  double *Tex;
  int i, it, num_time_steps, it_print;
  FILE* fp;  

  // read inputs
  fp = fopen("input.in", "r");
  fscanf(fp, "%d\n", &nx);
  fscanf(fp, "%lf %lf\n", &xst, &xen);
  fscanf(fp, "%lf %lf\n", &tst, &ten);
  fclose(fp);

  printf("Inputs are: %d %lf %lf %lf %lf\n", nx, xst, xen, tst, ten);

  x = (double *)malloc(nx*sizeof(double));
  T = (double *)malloc(nx*sizeof(double));
  Tex = (double *)malloc(nx*sizeof(double));
  rhs = (double *)malloc(nx*sizeof(double));
  

  grid(nx,xst,xen,x,&dx);         // initialize the grid

  set_initial_condition(nx,x,T);  // initial condition

  // prepare for time loop

  // time for Que 2 c) (a)
  dt = 1.25*1e-7;                               // Ensure r satisfies the stability condition
  
  // time for Que 2 c) (b)
  // dt = 3.125*1e-6;                               // Ensure r satisfies the stability condition
 
  num_time_steps = (int)((ten-tst)/dt) + 1; // why add 1 to this?
  it_print = num_time_steps/10;             // write out approximately 10 intermediate results

  // compute exact solution
  exactSolution(nx,ten,x,Tex);

  // start time stepping loop
  for(it=0; it<num_time_steps; it++)
  {
    tcurr = tst + (double)it * dt;

    timestep_Euler(nx,dt,dx,x,T,rhs);    // update T

    // output soln every it_print time steps
    if(it%it_print==0){
      output_soln(nx,it,tcurr,x,T,Tex);
      double l2err = get_l2err_norm(nx, T, Tex);
      printf("l2err at time step %d = %lf\n", it, l2err);
    }
      
  }

  free(rhs);
  free(T);
  free(Tex);
  free(x);

  return 0;
}

