#include <cuda_runtime.h>
#include <fstream>
#include <chrono>
#include "common.h"

#define DEBUG
//#undef DEBUG

extern void GenerateNextLevelParallel(std::map<MatchPoint, LeveledDAGNode*>& DAG, std::vector<LeveledDAGNode*>& currentLevel,
	std::vector<LeveledDAGNode*>& nextLevel, int sequenceCount, int* successorTableCuda, LeveledDAGNode* endNode);

extern void RemoveOutdatedNodeParallel(std::vector<LeveledDAGNode*>& noIncidentEdgeNode);

char** ReadSequenceFromFile(int number) {
	std::ifstream file = std::ifstream("DNA Sequence.txt", std::ios::in);
	if (!file.is_open()) {
		std::cout << "ERROR::Fail to load Sequence." << std::endl;
	}

	char** sequence = new char* [number];
	for (int i = 0; i < number; i++) {
		sequence[i] = new char[SEQUENCE_LENGTH];
	}

	for (int i = 0; i < number; i++) {
		char* temp = new char[SEQUENCE_LENGTH + 50];
		file.getline(temp, SEQUENCE_LENGTH + 50);
		memcpy(sequence[i], temp, SEQUENCE_LENGTH);
	}

	file.close();
	return sequence;
}

int* ConstructSuccessorTable(int sequenceCount, char** sequence) {
	// Allocate memory required for successor table
	int* successorTable = new int[sequenceCount * BASES_COUNT * (SEQUENCE_LENGTH + 1)];
	memset(successorTable, 0, sizeof(int) * sequenceCount * BASES_COUNT * (SEQUENCE_LENGTH + 1));

	// Fill the successor table
	for (int i = 0; i < sequenceCount; i++) {
		// Only scan the sequence once
		int pointer[BASES_COUNT] = { 0 };
		for (int j = 0; j < SEQUENCE_LENGTH; j++) {
			NitrogenousBases base = NitrogenousBases::A;
			switch (sequence[i][j]) {
			case 'a': base = NitrogenousBases::A; break;
			case 'g': base = NitrogenousBases::G; break;
			case 'c': base = NitrogenousBases::C; break;
			case 't': base = NitrogenousBases::T; break;
			}

			while (pointer[(int)base] <= j) {
				successorTable[i * BASES_COUNT * (SEQUENCE_LENGTH + 1) + (int)base * (SEQUENCE_LENGTH + 1) + pointer[(int)base]] = j + 1;
				pointer[(int)base]++;
			}
		}
	}

	return successorTable;
}

void RemoveOutdatedNode(std::map<MatchPoint, LeveledDAGNode*>& DAG, LeveledDAGNode* endNode) {
	std::vector<LeveledDAGNode*> noIncidentEdgeNode;
	for (auto it = DAG.begin(); it != DAG.end(); it++) {
		if (it->second->incidentEdgeCount == 0 && it->second != endNode) noIncidentEdgeNode.push_back(it->second);
	}

	RemoveOutdatedNodeParallel(noIncidentEdgeNode);

	for (int i = 0; i < noIncidentEdgeNode.size(); i++) {
		DAG.erase(noIncidentEdgeNode[i]->matchPoint);
		delete noIncidentEdgeNode[i];
	}
}

void SwapBetweenLevels(std::vector<LeveledDAGNode*>& currentLevel, std::vector<LeveledDAGNode*>& nextLevel) {
	currentLevel.swap(nextLevel);
	nextLevel.clear();
}

int main(int argc, char** argv) {
#ifdef DEBUG
	int sequenceCount = 5;
#else
	int sequenceCount = std::atoi(argv[1]);
#endif

	if (sequenceCount < 2 || sequenceCount > 10) {
		std::cout << "ERROR::Should be bigger than 3 and less or equal to 10." << std::endl;
	}

	char** sequence = ReadSequenceFromFile(sequenceCount);

	int* successorTable = ConstructSuccessorTable(sequenceCount, sequence);

	std::map<MatchPoint, LeveledDAGNode*> leveledDAG;
	std::vector<LeveledDAGNode*> currentLevel;
	std::vector<LeveledDAGNode*> nextLevel;

	// Initialized before constructing leveled DAG
	LeveledDAGNode* sourceNode = new LeveledDAGNode(sequenceCount, 0);
	LeveledDAGNode* endNode = new LeveledDAGNode(sequenceCount, std::numeric_limits<int>::max());
	leveledDAG.insert(std::make_pair(sourceNode->matchPoint, sourceNode));
	leveledDAG.insert(std::make_pair(endNode->matchPoint, endNode));
	currentLevel.push_back(sourceNode);

	int* successorTableCuda;
	cudaMalloc((void**)&successorTableCuda, sizeof(int) * sequenceCount * BASES_COUNT * (SEQUENCE_LENGTH + 1));
	cudaMemcpy(successorTableCuda, successorTable, sizeof(int) * sequenceCount * BASES_COUNT * (SEQUENCE_LENGTH + 1), cudaMemcpyHostToDevice);

	while (currentLevel.size() != 0) {
		GenerateNextLevelParallel(leveledDAG, currentLevel, nextLevel, sequenceCount, successorTableCuda, endNode);
		RemoveOutdatedNode(leveledDAG, endNode);
		SwapBetweenLevels(currentLevel, nextLevel);
	}

	while (leveledDAG.size() > 1) {
		RemoveOutdatedNode(leveledDAG, endNode);
	}

	endNode->PrintMLCS();
	std::cout << "Length: " << endNode->MLCSLength() << std::endl;

	return 0;
}