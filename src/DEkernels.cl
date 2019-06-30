/*	Parallel Random Number Generator in OpenCL
	Copyright (C) 2011 Giorgos Arampatzis, Angelos Athanasopoulos

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>. */

typedef struct
{
    uint aaa;
    int mm,nn,rr,ww;
    uint wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint maskB, maskC;
    int i;
}mt_struct_naked;

//*******************************************************************************
//****     Pseudo Random Number Generator       *********************************
//*******************************************************************************

typedef struct
{
    uint aaa;
    int mm,nn,rr,ww;
    uint wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint maskB, maskC;
    int i;
    uint state[MT_NN];
}mt_struct;

void sgenrand_mt(uint seed, mt_struct *mts) 
{
    int i;

    for (i=0; i < mts->nn; i++)
	{
		mts->state[i] = seed;
        seed = ( (1812433253U) * (seed  ^ (seed >> 30))) + i + 1;
    }//end for
    mts->i = mts->nn;

    for (i=0; i < mts->nn; i++)
	{
		mts->state[i] &= mts->wmask;
	}//end for
}

uint genrand_mt(mt_struct *mts) 
{
    uint *st, uuu, lll, aa, x;
    int k,n,m,lim;

    if ( mts->i >= mts->nn )
	{
		n = mts->nn; m = mts->mm;
		aa = mts->aaa;
		st = mts->state;
		uuu = mts->umask; lll = mts->lmask;

		lim = n - m;
		for (k=0; k < lim; k++)
		{
			x = (st[k]&uuu)|(st[k+1]&lll);
			st[k] = st[k+m] ^ (x>>1) ^ (x&1U ? aa : 0U);
		}//end for
		lim = n - 1;
		for (; k < lim; k++)
		{
			x = (st[k]&uuu)|(st[k+1]&lll);
			st[k] = st[k+m-n] ^ (x>>1) ^ (x&1U ? aa : 0U);
		}//end for
		x = (st[n-1]&uuu)|(st[0]&lll);
		st[n-1] = st[m-1] ^ (x>>1) ^ (x&1U ? aa : 0U);
		mts->i=0;
    }//end if
		
    x = mts->state[mts->i];
    mts->i += 1;
    x ^= x >> mts->shift0;
    x ^= (x << mts->shiftB) & mts->maskB;
    x ^= (x << mts->shiftC) & mts->maskC;
    x ^= x >> mts->shift1;

    return x;
}


float floatrand(mt_struct *mts)
{
	return ((float)genrand_mt(mts)) / ((float) 0xFFFFFFFF) ;
}

/*//genrand_mt must be 64 bit for double
double doublerand(mt_struct *mts)
{
	return ((double)genrand_mt(mts)) / ((double) 0xFFFFFFFFFFFFFFFF)) ;
}
*/


//*******************************************************************************
//***********       Functions  ******************************************
//*******************************************************************************

// Copy the data from an mt_struct_naked to an mt_struct
// The NG uses mt_struct structs.
void copyNakedToMts(mt_struct *mts, mt_struct_naked mts_naked)
{
	mts->aaa = mts_naked.aaa;
	mts->mm  = mts_naked.mm;
	mts->nn  = mts_naked.nn;
	mts->rr  = mts_naked.rr;
	mts->ww  = mts_naked.ww;
	mts->wmask  = mts_naked.wmask;
	mts->umask  = mts_naked.umask;
	mts->lmask  = mts_naked.lmask;
	mts->maskB  = mts_naked.maskB;
	mts->maskC  = mts_naked.maskC;
	mts->shift0  = mts_naked.shift0;
	mts->shift1  = mts_naked.shift1;
	mts->shiftB  = mts_naked.shiftB;
	mts->shiftC  = mts_naked.shiftC;
	mts->i  = mts_naked.i;
}

void copyNakedToMtsWithState(mt_struct *mts, mt_struct_naked mts_naked, __global uint *state)
{
	int i;
	int gid = get_global_id(0);

	mts->aaa = mts_naked.aaa;
	mts->mm  = mts_naked.mm;
	mts->nn  = mts_naked.nn;
	mts->rr  = mts_naked.rr;
	mts->ww  = mts_naked.ww;
	mts->wmask  = mts_naked.wmask;
	mts->umask  = mts_naked.umask;
	mts->lmask  = mts_naked.lmask;
	mts->maskB  = mts_naked.maskB;
	mts->maskC  = mts_naked.maskC;
	mts->shift0  = mts_naked.shift0;
	mts->shift1  = mts_naked.shift1;
	mts->shiftB  = mts_naked.shiftB;
	mts->shiftC  = mts_naked.shiftC;
	mts->i  = mts_naked.i;
	
	for(i=0; i < MT_NN; i++)
	{
		mts->state[i] = state[MT_NN*gid+i];
	}//end for
}

void SaveState(__global uint *state, __global mt_struct_naked *mts_naked, mt_struct *mts)
{
	int i;
	int gid = get_global_id(0);
	for(i=0; i < MT_NN; i++)
	{
		state[MT_NN*gid+i] = mts->state[i];
	}//end for
	mts_naked[gid].i = mts->i;
}

/*
//Tried to calculate fitness per work item
double evaluateFitness(__global double* trialVector)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int groupId = get_group_id(0);
	
	sum = 0;
	
	localSquare[lid] = trialVector[lid] * trialVector[lid];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
    if(lid == 0)
	{
		for(int counter = 0; counter < get_local_size(0); counter++)
		{
			sum += localSquare[counter];
		}//end for
    }//end if
	return sum;
}
*/

//*******************************************************************************
//*********** Compute Kernel Functions  *****************************************
//*******************************************************************************

__kernel void initRNG(	__global mt_struct_naked *mts_naked,
						__global uint *state,
						__constant int *seeds)
{
	__private int gid = get_global_id(0);
	mt_struct mts;

	copyNakedToMts(&mts, mts_naked[gid]);
	sgenrand_mt(seeds[gid], &mts);

	SaveState(state, mts_naked, &mts);
}//end initRNG

//*******************************************************************************
//*********** DE Kernels ********************************************************
//*******************************************************************************

__kernel void createPopulationSphere(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation)
{
	mt_struct mts;
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__local double localSquare[D];// D <=> local_size();
	
	fitnessOfPopulation[groupId] = 0;	
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	// and evaluate fitness function
	localSquare[lid] = trialVector[lid] * trialVector[lid];
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lid == 0)
	{
		for(__private int counter = 0; counter < get_local_size(0); counter++)
		{
			fitnessOfPopulation[groupId] += localSquare[counter];
		}//end for
	}//end if
	
	SaveState(state, mts_naked, &mts);
}//end createPopulationSphere

__kernel void createPopulationAckley(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation)
{
	mt_struct mts;
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__private const double c = 2 * M_PI;
	__private const double b = 0.2;
	__private const double a = 20;
	__local double localSquare[D], localCosine[D];
	__local double sum1, sum2, term1, term2;// D <=> local_size();
	
	sum1 = 0, sum2 = 0;	
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	// and evaluate fitness function
	localSquare[lid] = trialVector[lid] * trialVector[lid];
	localCosine[lid] = cos(c * trialVector[lid]);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lid == 0)
	{
		for(__private int counter = 0; counter < get_local_size(0); counter++)
		{
			sum1 += localSquare[counter];
			sum2 += localCosine[counter];
		}//end for
	}//end if
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	term1 = -a * exp(-b * sqrt(sum1 / get_local_size(0)));
	term2 = -exp(sum2 / get_local_size(0));
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	fitnessOfPopulation[groupId] = term1 + term2 + a + exp((double)1);

	SaveState(state, mts_naked, &mts);
}//end createPopulationAckley

__kernel void createPopulationRosenbrock(	__global mt_struct_naked *mts_naked,
											__global uint *state,
											__global double *population,
											__global double *trialVector,
											__global double *fitnessOfPopulation)
{
	mt_struct mts;
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__local double localRosen[D];// D <=> local_size();
	
	fitnessOfPopulation[groupId] = 0;	
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	localRosen[lid] = 100 * pow((trialVector[lid + 1] - pow(trialVector[lid], 2)), 2) + pow(trialVector[lid] - 1, 2);
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(lid == 0)
	{
		for(__private int counter = 0; counter < get_local_size(0); counter++)
		{
			fitnessOfPopulation[groupId] += localRosen[counter];
		}//end for
	}//end if
	
	SaveState(state, mts_naked, &mts);
}//end createPopulationRosenbrock

__kernel void createPopulationSchaffer(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation)
{
	mt_struct mts;
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	__private double x1 = trialVector[0], x2 = trialVector[1];
	
	localFact1[groupId] = pow(sin(pow(x1, 2) - pow(x2, 2)), 2) - 0.5;
	localFact2[groupId] = pow(1 + 0.001 * (pow(x1, 2) + pow(x2, 2)), 2);
		
	fitnessOfPopulation[groupId] = 0.5 + localFact1[groupId] / localFact2[groupId];
	
	SaveState(state, mts_naked, &mts);
}//end createPopulationSchaffer

__kernel void createPopulationEggHolder(__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation)
{
	mt_struct mts; 
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	__private double x1 = trialVector[0], x2 = trialVector[1];
	
	localFact1[groupId] = -(x2 + 47) * sin(sqrt(fabs(x2 + (x1 / 2) + 47)));
	localFact2[groupId] = -x1 * sin(sqrt(fabs(x1 - (x2 + 47))));
	
	fitnessOfPopulation[groupId] = localFact1[groupId] + localFact2[groupId];
	
	SaveState(state, mts_naked, &mts);
}//end createPopulationEggHolder

__kernel void createPopulationHolderTable(	__global mt_struct_naked *mts_naked,
											__global uint *state,
											__global double *population,
											__global double *trialVector,
											__global double *fitnessOfPopulation)
{
	mt_struct mts; 
	__private int gid = get_global_id(0);//index
	__private int lid = get_local_id(0);//column index
	__private int groupId = get_group_id(0);//row index
	//O.F. variables
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
	
	copyNakedToMtsWithState(&mts, mts_naked[gid], state);
	
	// initialize each individual
	// within boundary constraints
	population[gid] = trialVector[lid] =  L + (H-L) * floatrand(&mts);
	
	__private double x1 = trialVector[0], x2 = trialVector[1];
	
	localFact1[groupId] = sin(x1) * cos(x2);
	localFact2[groupId] = exp(fabs(1 - sqrt(pow(x1, 2) + pow(x2, 2)) / M_PI));
	
	fitnessOfPopulation[groupId] = -fabs(localFact1[groupId] * localFact2[groupId]);
	
	SaveState(state, mts_naked, &mts);
}//end createPopulationHolderTable

__kernel void optimizationSphere(	__global mt_struct_naked *mts_naked,
									__global uint *state,
									__global double *population,
									__global double *trialVector,
									__global double *fitnessOfPopulation,
									__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	__local double localSquare[D];// D <=> local_size();
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		fitnessOfNewIndividual[groupId] = 0;
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		localSquare[lid] = trialVector[lid] * trialVector[lid];
		
		barrier(CLK_LOCAL_MEM_FENCE);
			
		if(lid == 0)
		{
			for(__private int counter = 0; counter < get_local_size(0); counter++)
			{
				fitnessOfNewIndividual[groupId] += localSquare[counter];
			}//end for
		}//end if
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationSphere

__kernel void optimizationAckley(	__global mt_struct_naked *mts_naked,
									__global uint *state,
									__global double *population,
									__global double *trialVector,
									__global double *fitnessOfPopulation,
									__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	//O.F. variables
	__private const double c = 2 * M_PI;
	__private const double b = 0.2;
	__private const double a = 20;
	__local double localSquare[D], localCosine[D];
	__local double sum1, sum2, term1, term2;// D <=> local_size();
	
	sum1 = 0, sum2 = 0;
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		localSquare[lid] = trialVector[lid] * trialVector[lid];
		localCosine[lid] = cos(c * trialVector[lid]);
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lid == 0)
		{
			for(__private int counter = 0; counter < get_local_size(0); counter++)
			{
				sum1 += localSquare[counter];
				sum2 += localCosine[counter];
			}//end for
		}//end if
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		term1 = -a * exp(-b * sqrt(sum1 / get_local_size(0)));
		term2 = -exp(sum2 / get_local_size(0));
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		fitnessOfNewIndividual[groupId] = term1 + term2 + a + exp((double)1);
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationAckley

__kernel void optimizationRosenbrock(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation,
										__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	__local double localRosen[D];// D <=> local_size();
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		fitnessOfNewIndividual[groupId] = 0;
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		localRosen[lid] = 100 * pow((trialVector[lid + 1] - pow(trialVector[lid], 2)), 2) + pow(trialVector[lid] - 1, 2);
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(lid == 0)
		{
			for(__private int counter = 0; counter < get_local_size(0); counter++)
			{
				fitnessOfNewIndividual[groupId] += localRosen[counter];
			}//end for
		}//end if
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationRosenbrock

__kernel void optimizationSchaffer(	__global mt_struct_naked *mts_naked,
									__global uint *state,
									__global double *population,
									__global double *trialVector,
									__global double *fitnessOfPopulation,
									__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		__private double x1 = trialVector[0], x2 = trialVector[1];
		
		localFact1[groupId] = pow(sin(pow(x1, 2) - pow(x2, 2)), 2) - 0.5;
		localFact2[groupId] = pow(1 + 0.001 * (pow(x1, 2) + pow(x2, 2)), 2);
			
		fitnessOfNewIndividual[groupId] = 0.5 + localFact1[groupId] / localFact2[groupId];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationSchaffer

__kernel void optimizationEggHolder(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation,
										__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		__private double x1 = trialVector[0], x2 = trialVector[1];
		
		localFact1[groupId] = -(x2 + 47) * sin(sqrt(fabs(x2 + (x1 / 2) + 47)));
		localFact2[groupId] = -x1 * sin(sqrt(fabs(x1 - (x2 + 47))));
			
		fitnessOfNewIndividual[groupId] = localFact1[groupId] + localFact2[groupId];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationEggHolder

__kernel void optimizationHolderTable(	__global mt_struct_naked *mts_naked,
										__global uint *state,
										__global double *population,
										__global double *trialVector,
										__global double *fitnessOfPopulation,
										__global double *fitnessOfNewIndividual)
{
	__private int gid = get_global_id(0);
	__private int lid = get_local_id(0);
	__private int groupId = get_group_id(0);
	__local double localFact1[NP], localFact2[NP];// D <=> local_size();
		
	for (__private int g = 0; g < GEN; g++)
	{
		mt_struct mts; 
		copyNakedToMtsWithState(&mts, mts_naked[gid], state);

		__private int rdmIndices[3], index[3], mutationParameter;
		
		// choose three random individuals from population,
        // mutually different and also different from j
		for (__private int counter = 0; counter < 3; counter++)
		{
			rdmIndices[counter] = (int)(get_num_groups(0) * floatrand(&mts));
			index[counter] = lid + get_local_size(0) * rdmIndices[counter];
		}//end for
		
		// create trial individual
        // in which at least one parameter is changed
		mutationParameter = (int)(get_local_size(0) * floatrand(&mts));
		
		if(floatrand(&mts) < CR || mutationParameter == lid)
		{
			trialVector[lid] = population[index[2]] + F * (population[index[0]] - population[index[1]]);
		}//end if
		else
		{
			trialVector[lid] = population[gid];
		}//end if
		
		// verify boundary constraints
		if ((trialVector[lid] < L) || (trialVector[lid] > H))
		{
			trialVector[lid] = L + (H-L) * floatrand(&mts);
		}//end if
		
		// select the best individual
        // between trial and current ones
		//evaluate Fitness of trial individual(s)
		__private double x1 = trialVector[0], x2 = trialVector[1];
		
		localFact1[groupId] = sin(x1) * cos(x2);
		localFact2[groupId] = exp(fabs(1 - sqrt(pow(x1, 2) + pow(x2, 2)) / M_PI));
			
		fitnessOfNewIndividual[groupId] = -fabs(localFact1[groupId] * localFact2[groupId]);
		
		barrier(CLK_LOCAL_MEM_FENCE);
		
		// if trial is better or equal than current
		if (fitnessOfNewIndividual[groupId] <= fitnessOfPopulation[groupId])
		{
			// replace current by trial
			population[gid] = trialVector[lid];
			fitnessOfPopulation[groupId] = fitnessOfNewIndividual[groupId];			
		}//end if
		SaveState(state, mts_naked, &mts);
	}//end for
}//end optimizationHolderTable