/*
compile with the following two commands from command line (Linux only):
g++ -c -Wall -Werror -O3 -fpic fast_optimizer.cpp
g++ -shared -Wall -O3 -o fast_optimizer.so fast_optimizer.o
The flags -Wall -O3 can technically be omitted, but should not!
*/

#include <math.h>
#include <iostream>
using namespace std;

void optimize(int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double EL[][3], double XX[], double al){
    double tol = 1e-9;

    double B[NUMSITES][3],C[NUMSITES][DIMENSION];
    int eqnstack[NUMSITES], leafQ[NUMSITES], val[NUMSITES], intern_edge[NUMSITES][3];
    int i,m,j,i2;
    int n0,n1,n2,lqp,eqp,k1;
    double q0,q1,q2,t;

    lqp = eqp = 0;
    k1 = NUMSITES-2;

    /* prepare equations */ 
    for(i=k1-1;i>=0;i--){
        n0 = adj[i][0];
        n1 = adj[i][1];
        n2 = adj[i][2];
        q0 = pow(fabs(EW[i][0]),al)/(EL[i][0]+tol);
        q1 = pow(fabs(EW[i][1]),al)/(EL[i][1]+tol);
        q2 = pow(fabs(EW[i][2]),al)/(EL[i][2]+tol);

        t = q0+q1+q2;
        q0 /= t;
        q1 /= t;
        q2 /= t;

        val[i] = 0;
        B[i][0] = B[i][1] = B[i][2] = 0.0;
        intern_edge[i][0] = intern_edge[i][1] = intern_edge[i][2] = 0;

        for(m=0;m<DIMENSION;m++){C[i][m] = 0.0;}

        #define prep(a,b,c) if(b>=NUMSITES){val[i]++;B[i][a]=c;intern_edge[i][a]=1;}else{for(m=0;m<DIMENSION;m++){C[i][m]+=XX[b*DIMENSION+m]*c;}}
        prep(0,n0,q0);
        prep(1,n1,q1);
        prep(2,n2,q2);

        if(val[i]<=1){leafQ[lqp]=i,lqp++;}
    }
    while(lqp > 1){ 
        /* eliminate leaf i from tree*/ 
        lqp--; i = leafQ[lqp]; val[i]--; i2 = i+ NUMSITES; 
        eqnstack[eqp] = i; eqp++;/* push i in stack */ 
        for(j =0; j < 3; j++){if(intern_edge[i][j] != 0){break;}}
        q0 = B[i][j]; 
        j = adj[i][j]-NUMSITES;/* neighbor is j */ 
        val[j]-- ; 
        if(val[j] == 1){ leafQ[lqp] = j; lqp ++; }/* check if neighbor has become leaf? */ 
        for(m=0; m<3; m++){if(adj[j][m] == i2){break;}} 
        q1 = B[j][m]; B[j][m] = 0.0; intern_edge[j][m] = 0; 
        t = 1.0-q1*q0; t = 1.0/t; 
        for(m=0; m<3; m++){B[j][m] *= t;} 
        for(m=0; m<DIMENSION; m++){ C[j][m] += q1*C[i][m]; C[j][m] *= t; } 
    }
    /* Solve trivial tree */ 
    i = leafQ[0]; i2 = i + NUMSITES; 
    for(m=0; m <DIMENSION; m++){XX[i2*DIMENSION+m] = C[i][m];}
    /* Solve rest by backsolving */ 
    while(eqp > 0){ 
        eqp--; i = eqnstack[eqp]; i2 = i+ NUMSITES; 
        for(j =0; j <3; j ++){if(intern_edge[i][j] != 0){break;}}/* find neighbor j */ 
        q0 = B[i][j]; 
        j = adj[i][j];/* get neighbor indeces */ 
        for(m = 0; m < DIMENSION; m++){XX[i2*DIMENSION+m] = C[i][m] + q0*XX[j*DIMENSION+m];}
    }    
    return;
}


double length(int DIMENSION, int NUMSITES, int adj[][3],  double EW[][3], double EL[][3], double XX[], double al) {
    /*calculates the cost of the current configuration and stores edge lengths in EL*/
    #define dist(a,b) t=0.0;for(m=0;m<DIMENSION;m++){r=XX[a*DIMENSION+m]-XX[b*DIMENSION+m];t+=r*r;}t=sqrt(t);
    int m,i2,i,j;
    int n0,n1,n2,k1;
    double leng,t,r;
    leng = 0.0;
    k1=NUMSITES-2;
    for(i=0;i<k1;i++){
        i2 = i+NUMSITES;
        n0 = adj[i][0];n1=adj[i][1];n2=adj[i][2];
        if(n0<i2){
            dist(n0,i2);leng+=pow(fabs(EW[i][0]),al)*t;EL[i][0]=t;n0-=NUMSITES;
            if(n0>=0)for(j=0;j<3;j++)if(adj[n0][j]==i2){EL[n0][j]=t;break;}
        }
        if(n1<i2){
            dist(n1,i2);leng+=pow(fabs(EW[i][1]),al)*t;EL[i][1]=t;n1-=NUMSITES;
            if(n1>=0)for(j=0;j<3;j++)if(adj[n1][j]==i2){EL[n1][j]=t;break;}
        }
        if(n2<i2){
            dist(n2,i2);leng+=pow(fabs(EW[i][2]),al)*t;EL[i][2]=t;n2-=NUMSITES;
            if(n2>=0)for(j=0;j<3;j++)if(adj[n2][j]==i2){EL[n2][j]=t;break;}
        }
    }
    return leng;
}

void calculate_EW(int NUMSITES, double EW[][3], int adj[][3], double demands[], double al){
    int i,j,m,i2;
    int n0,n1,n2;
    int leafQ[NUMSITES],val[NUMSITES];
    int lqp = 0;
    int done_edges[NUMSITES][3];
    double d[NUMSITES];

    for(i=0;i<NUMSITES-2;i++){
        n0 = adj[i][0];
        n1 = adj[i][1];
        n2 = adj[i][2];

        d[i] = 0.0;
        done_edges[i][0] = done_edges[i][1] = done_edges[i][2] = 0;
        EW[i][0] = EW[i][1] = EW[i][2] = 0.0;

        val[i] = 0;
        if(n0>=NUMSITES){val[i]++;}else{d[i]+=demands[n0];EW[i][0]=-demands[n0];done_edges[i][0]=1;}
        if(n1>=NUMSITES){val[i]++;}else{d[i]+=demands[n1];EW[i][1]=-demands[n1];done_edges[i][1]=1;}
        if(n2>=NUMSITES){val[i]++;}else{d[i]+=demands[n2];EW[i][2]=-demands[n2];done_edges[i][2]=1;}
        if(val[i]==1){leafQ[lqp]=i;lqp++;}
    }
    while(lqp>1){
        lqp--;
        i = leafQ[lqp]; i2=i+NUMSITES; val[i]--;
        for(m=0;m<3;m++){if(done_edges[i][m]==0){break;}} /*find parent BP*/
        EW[i][m] = d[i];
        done_edges[i][m] = 1;

        j = adj[i][m]-NUMSITES;
        for(m=0;m<3;m++){if(adj[j][m]==i2){break;}}
        EW[j][m] = -d[i];
        done_edges[j][m] = 1;
        d[j]+=d[i];
        val[j]--;
        if(val[j] == 1){ leafQ[lqp] = j; lqp ++; }/* check if neighbor has become leaf? */
    }
    //for(i=0;i<NUMSITES-2;i++){
    //    for(m=0;m<3;m++){EW[i][m] = pow(fabs(EW[i][m]),al);}/*turn flows into edge_weights*/
    //}
}

extern "C"
double iterations(int *iter, int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double demands[], double XX[], double al, double improv_thres = 1e-7){
    /*iteratively optimizes the BP configuration until improvement threshold is reached*/
    double cost,cost_old,improv;

    double EL[NUMSITES][3];

    calculate_EW(NUMSITES,EW,adj,demands,al);

    cost_old = length(DIMENSION, NUMSITES, adj, EW, EL, XX, al);
    improv = 1.0;
    *iter = 0;
    do{
        (*iter)++;
        optimize(DIMENSION, NUMSITES, adj, EW, EL, XX, al);
        cost=length(DIMENSION, NUMSITES, adj, EW, EL, XX, al);
        improv = cost_old - cost;
        cost_old =  cost;
    }while(improv>improv_thres);
    return cost;
}