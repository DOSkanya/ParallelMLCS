#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "common.h"
namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 1024
#define THREADS_PER_SM 32
#define MAX_SEQUENCE_COUNT 10

__global__ void GenerateNextLevelCuda(int* sequenceCountCuda, int* successorTableCuda, int* currentMatchPointsCuda, int* currentLevelSizeCuda, int* generatedSuccessorCuda) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= *currentLevelSizeCuda) return;
	
	__shared__ int currentMatchPointsShared[THREADS_PER_SM * MAX_SEQUENCE_COUNT];
	// TODO:  π”√shared memory

	for (int i = 0; i < BASES_COUNT; i++) {
		for (int j = 0; j < *sequenceCountCuda; j++) {
			generatedSuccessorCuda[index * BASES_COUNT * *sequenceCountCuda + i * *sequenceCountCuda + j]
				= successorTableCuda[j * BASES_COUNT * (SEQUENCE_LENGTH + 1) + i * (SEQUENCE_LENGTH + 1) + currentMatchPointsCuda[index * *sequenceCountCuda + j]];
		}
	}
}

__global__ void RemoveOutdatedNodeCuda(int* noIncidentEdgeNodeSizeCuda, int* partialLCSLengthCuda, int* numOfPartialLCSCuda, int* indexPrefixSumCuda, char* partialLCSBufferCuda, char* producedPartialLCSBufferCuda) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= *noIncidentEdgeNodeSizeCuda) return;

	char symbol[4] = {'a', 'g', 'c', 't'};

	int startPosition = indexPrefixSumCuda[index] * SEQUENCE_LENGTH;
	int bufferSize = numOfPartialLCSCuda[index] * SEQUENCE_LENGTH;
	int partialLCSLength = partialLCSLengthCuda[index];
	for (int i = 0; i < numOfPartialLCSCuda[index]; i++) {
		for (int j = 0; j < BASES_COUNT; j++) {
			memcpy(&producedPartialLCSBufferCuda[BASES_COUNT * startPosition + j * bufferSize + i * SEQUENCE_LENGTH], &partialLCSBufferCuda[startPosition + i * SEQUENCE_LENGTH], sizeof(char) * SEQUENCE_LENGTH);
			producedPartialLCSBufferCuda[BASES_COUNT * startPosition + j * bufferSize + i * SEQUENCE_LENGTH + partialLCSLength] = symbol[j];
		}
	}
}

void GenerateNextLevelParallel(std::map<MatchPoint, LeveledDAGNode*>& DAG, std::vector<LeveledDAGNode*>& currentLevel,
	std::vector<LeveledDAGNode*>& nextLevel, int sequenceCount, int* successorTableCuda, LeveledDAGNode* endNode) {
	int currentLevelSize = currentLevel.size();
	int* currentLevelSizeCuda;
	cudaMalloc((void**)&currentLevelSizeCuda, sizeof(int));
	cudaMemcpy(currentLevelSizeCuda, &currentLevelSize, sizeof(int), cudaMemcpyHostToDevice);

	int* currentMatchPoints = new int[sequenceCount * currentLevelSize];
	int* currentMatchPointsCuda;
	for (int i = 0; i < currentLevelSize; i++) {
		memcpy(&currentMatchPoints[sequenceCount * i], currentLevel[i]->matchPoint.positions, sizeof(int) * sequenceCount);
	}
	cudaMalloc((void**)&currentMatchPointsCuda, sizeof(int) * sequenceCount * currentLevelSize);
	cudaMemcpy(currentMatchPointsCuda, currentMatchPoints, sizeof(int) * sequenceCount * currentLevelSize, cudaMemcpyHostToDevice);

	int* generatedSuccessor = new int[BASES_COUNT * sequenceCount * currentLevelSize];
	int* generatedSuccessorCuda;
	cudaMalloc((void**)&generatedSuccessorCuda, sizeof(int) * BASES_COUNT * sequenceCount * currentLevelSize);

	int* sequenceCountCuda;
	cudaMalloc((void**)&sequenceCountCuda, sizeof(int));
	cudaMemcpy(sequenceCountCuda, &sequenceCount, sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockDim((currentLevelSize / THREADS_PER_BLOCK) + 1, 1, 1);
	dim3 threadDim(THREADS_PER_BLOCK, 1, 1);
	GenerateNextLevelCuda<<<blockDim, threadDim>>>(sequenceCountCuda, successorTableCuda, currentMatchPointsCuda, currentLevelSizeCuda, generatedSuccessorCuda);
	cudaDeviceSynchronize();

	cudaMemcpy(generatedSuccessor, generatedSuccessorCuda, sizeof(int) * BASES_COUNT * sequenceCount * currentLevelSize, cudaMemcpyDeviceToHost);

	cudaFree(currentLevelSizeCuda);
	cudaFree(currentMatchPointsCuda);
	cudaFree(generatedSuccessorCuda);
	cudaFree(sequenceCountCuda);

	for (int i = 0; i < currentLevelSize; i++) {
		for (int j = 0; j < BASES_COUNT; j++) {
			MatchPoint newMatchPoint(sequenceCount, 0);
			memcpy(newMatchPoint.positions, &generatedSuccessor[i * BASES_COUNT * sequenceCount + j * sequenceCount], sizeof(int) * sequenceCount);
			LeveledDAGNode* currentLevelNode = currentLevel[i];
			if (newMatchPoint.isValid()) {
				auto it = DAG.find(newMatchPoint);
				if (it == DAG.end()) {
					LeveledDAGNode* newDAGNode = new LeveledDAGNode(sequenceCount, 0);
					newDAGNode->matchPoint = newMatchPoint;
					char* LCS = new char[SEQUENCE_LENGTH];
					switch (j) {
					case 0: LCS[0] = 'a'; newDAGNode->symbol = NitrogenousBases::A; break;
					case 1: LCS[0] = 'g'; newDAGNode->symbol = NitrogenousBases::G; break;
					case 2: LCS[0] = 'c'; newDAGNode->symbol = NitrogenousBases::C; break;
					case 3: LCS[0] = 't'; newDAGNode->symbol = NitrogenousBases::T; break;
					}
					newDAGNode->partialLCS.push_back(LCS);
					newDAGNode->releaseHeader.push_back(LCS);
					newDAGNode->incidentEdgeCount++;
					newDAGNode->partialLCSLength++;
					DAG.insert(std::make_pair(newDAGNode->matchPoint, newDAGNode));
					currentLevelNode->successors.push_back(newDAGNode);
					nextLevel.push_back(newDAGNode);
				}
				else {
					delete[] newMatchPoint.positions;
					currentLevelNode->successors.push_back(it->second);
					it->second->incidentEdgeCount++;
				}
			}

			if (currentLevelNode->successors.size() == 0) {
				currentLevelNode->successors.push_back(endNode);
				endNode->incidentEdgeCount++;
			}
		}
	}

	delete[] currentMatchPoints;
	delete[] generatedSuccessor;
}

void RemoveOutdatedNodeParallel(std::vector<LeveledDAGNode*>& noIncidentEdgeNode) {
	int noIncidentEdgeNodeSize = noIncidentEdgeNode.size();
	/*int* noIncidentEdgeNodeSizeCuda;
	cudaMalloc((void**)&noIncidentEdgeNodeSizeCuda, sizeof(int));
	cudaMemcpy(noIncidentEdgeNodeSizeCuda, &noIncidentEdgeNodeSize, sizeof(int), cudaMemcpyHostToDevice);*/

	int* partialLCSLength = new int[noIncidentEdgeNodeSize] {0};
	//int* partialLCSLengthCuda;
	for (int i = 0; i < noIncidentEdgeNodeSize; i++) partialLCSLength[i] = noIncidentEdgeNode[i]->partialLCSLength;
	//cudaMalloc((void**)&partialLCSLengthCuda, sizeof(int) * noIncidentEdgeNodeSize);
	//cudaMemcpy(partialLCSLengthCuda, partialLCSLength, sizeof(int) * noIncidentEdgeNodeSize, cudaMemcpyHostToDevice);*/

	int* numOfPartialLCS = new int[noIncidentEdgeNodeSize] {0};
	//int* numOfPartialLCSCuda;
	for (int i = 0; i < noIncidentEdgeNodeSize; i++) numOfPartialLCS[i] = noIncidentEdgeNode[i]->partialLCS.size();
	//cudaMalloc((void**)&numOfPartialLCSCuda, sizeof(int) * noIncidentEdgeNodeSize);
	//cudaMemcpy(numOfPartialLCSCuda, numOfPartialLCS, sizeof(int) * noIncidentEdgeNodeSize, cudaMemcpyHostToDevice);*/

	int* indexPrefixSum = new int[noIncidentEdgeNodeSize] {0};
	//int* indexPrefixSumCuda;
	for (int i = 1; i < noIncidentEdgeNodeSize; i++) {
		indexPrefixSum[i] = indexPrefixSum[i - 1] + numOfPartialLCS[i - 1];
	}
	//cudaMalloc((void**)&indexPrefixSumCuda, sizeof(int) * noIncidentEdgeNodeSize);
	//cudaMemcpy(indexPrefixSumCuda, indexPrefixSum, sizeof(int) * noIncidentEdgeNodeSize, cudaMemcpyHostToDevice);*/

	int totalNumOfPartialLCS = indexPrefixSum[noIncidentEdgeNodeSize - 1] + numOfPartialLCS[noIncidentEdgeNodeSize - 1];

	char* partialLCSBuffer = new char[totalNumOfPartialLCS * SEQUENCE_LENGTH] {0};
	//char* partialLCSBufferCuda;
	for (int i = 0; i < noIncidentEdgeNodeSize; i++) {
		for (int j = 0; j < noIncidentEdgeNode[i]->partialLCS.size(); j++) {
			int storePosition = (indexPrefixSum[i] + j) * SEQUENCE_LENGTH;
			memcpy(&partialLCSBuffer[storePosition], noIncidentEdgeNode[i]->partialLCS[j], sizeof(char) * SEQUENCE_LENGTH);
		}
	}
	//cudaMalloc((void**)&partialLCSBufferCuda, sizeof(char) * totalNumOfPartialLCS * SEQUENCE_LENGTH);
	//cudaMemcpy(partialLCSBufferCuda, partialLCSBuffer, sizeof(char) * totalNumOfPartialLCS * SEQUENCE_LENGTH, cudaMemcpyHostToDevice);*/

	/*char* producedPartialLCSBuffer = new char[BASES_COUNT * totalNumOfPartialLCS * SEQUENCE_LENGTH] {0};
	char* producedPartialLCSBufferCuda;
	cudaMalloc((void**)&producedPartialLCSBufferCuda, sizeof(char) * BASES_COUNT * totalNumOfPartialLCS * SEQUENCE_LENGTH);
	
	dim3 blockDim((noIncidentEdgeNodeSize / THREADS_PER_BLOCK) + 1, 1, 1);
	dim3 threadDim(THREADS_PER_BLOCK, 1, 1);*/
	//RemoveOutdatedNodeCuda<<<blockDim, threadDim>>>(noIncidentEdgeNodeSizeCuda, partialLCSLengthCuda, numOfPartialLCSCuda, indexPrefixSumCuda, partialLCSBufferCuda, producedPartialLCSBufferCuda);
	//cudaDeviceSynchronize();

	//cudaMemcpy(producedPartialLCSBuffer, producedPartialLCSBufferCuda, sizeof(char) * BASES_COUNT * totalNumOfPartialLCS * SEQUENCE_LENGTH, cudaMemcpyDeviceToHost);

	/*cudaFree(noIncidentEdgeNodeSizeCuda);
	cudaFree(partialLCSLengthCuda);
	cudaFree(numOfPartialLCSCuda);
	cudaFree(indexPrefixSumCuda);
	cudaFree(partialLCSBufferCuda);*/
	//cudaFree(producedPartialLCSBufferCuda);

	char symbols[4] = { 'a', 'g', 'c', 't' };

	for (int i = 0; i < noIncidentEdgeNodeSize; i++) {
		LeveledDAGNode* node = noIncidentEdgeNode[i];
		for (int j = 0; j < node->successors.size(); j++) {
			int s_length = node->successors[j]->partialLCSLength;
			int p_length = node->partialLCSLength;

			if (p_length >= s_length) {
				node->vectorRelease(node->successors[j]->releaseHeader);
				node->successors[j]->releaseHeader.clear();
				node->successors[j]->releaseHeader.shrink_to_fit();
				node->successors[j]->partialLCS.clear();
				node->successors[j]->partialLCS.shrink_to_fit();
				char* strAllInOne = new char[numOfPartialLCS[i] * SEQUENCE_LENGTH];
				int symbol = (int)node->successors[j]->symbol;
				/*int loadPosition = BASES_COUNT * indexPrefixSum[i] * SEQUENCE_LENGTH + symbol * numOfPartialLCS[i] * SEQUENCE_LENGTH;
				memcpy(strAllInOne, &producedPartialLCSBuffer[loadPosition], sizeof(char) * numOfPartialLCS[i] * SEQUENCE_LENGTH);*/

				int loadPosition = indexPrefixSum[i] * SEQUENCE_LENGTH;
				memcpy(strAllInOne, &partialLCSBuffer[loadPosition], sizeof(char) * numOfPartialLCS[i] * SEQUENCE_LENGTH);
				for (int k = 0; k < numOfPartialLCS[i]; k++) {
					strAllInOne[k * SEQUENCE_LENGTH + partialLCSLength[i]] = symbols[symbol];
					node->successors[j]->partialLCS.push_back(&strAllInOne[k * SEQUENCE_LENGTH]);
				}
				node->successors[j]->partialLCSLength = node->partialLCSLength + 1;
				node->successors[j]->releaseHeader.push_back(strAllInOne);
			}
			else if (p_length + 1 == s_length && p_length != 0) {
				char* strAllInOne = new char[numOfPartialLCS[i] * SEQUENCE_LENGTH];
				int symbol = (int)node->successors[j]->symbol;
				/*int loadPosition = BASES_COUNT * indexPrefixSum[i] * SEQUENCE_LENGTH + symbol * numOfPartialLCS[i] * SEQUENCE_LENGTH;
				memcpy(strAllInOne, &producedPartialLCSBuffer[loadPosition], sizeof(char) * numOfPartialLCS[i] * SEQUENCE_LENGTH);*/
				int loadPosition = indexPrefixSum[i] * SEQUENCE_LENGTH;
				memcpy(strAllInOne, &partialLCSBuffer[loadPosition], sizeof(char) * numOfPartialLCS[i] * SEQUENCE_LENGTH);
				for (int k = 0; k < numOfPartialLCS[i]; k++) {
					strAllInOne[k * SEQUENCE_LENGTH + partialLCSLength[i]] = symbols[symbol];
					node->successors[j]->partialLCS.push_back(&strAllInOne[k * SEQUENCE_LENGTH]);
				}
				node->successors[j]->releaseHeader.push_back(strAllInOne);
			}

			node->successors[j]->incidentEdgeCount--;
		}
	}

	delete[] partialLCSLength;
	delete[] numOfPartialLCS;
	delete[] indexPrefixSum;
	if (totalNumOfPartialLCS > 0) {
		delete[] partialLCSBuffer;
		//delete[] producedPartialLCSBuffer;
	}
}