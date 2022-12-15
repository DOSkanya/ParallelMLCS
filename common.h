#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <string>


#define SEQUENCE_LENGTH 120
#define BASES_COUNT 4

enum class NitrogenousBases
{
	A, G, C, T
};

class MatchPoint {
public:
	MatchPoint(int matchPointLength, int initValue) {
		length = matchPointLength;
		positions = new int[length];
		if (initValue == 0) memset(positions, 0, sizeof(int) * length);
		else {
			for (int i = 0; i < length; i++) positions[i] = initValue;
		}
	}

	MatchPoint(MatchPoint& mp, int index, int* successorTable) {
		length = mp.length;
		positions = new int[length];
		memset(positions, 0, sizeof(int) * length);
		for (int i = 0; i < length; i++) {
			positions[i] = successorTable[i * BASES_COUNT * (SEQUENCE_LENGTH + 1) + index * (SEQUENCE_LENGTH + 1) + mp.positions[i]];
		}
	}

	bool isValid() {
		for (int i = 0; i < length; i++) {
			if (positions[i] == 0) return false;
		}
		return true;
	}

	bool operator < (const MatchPoint& mp) const {
		for (int i = 0; i < this->length; i++) {
			if (mp.positions[i] > this->positions[i]) return true;
			else if (mp.positions[i] < this->positions[i]) return false;
			else if (i == this->length - 1 && mp.positions[i] == this->positions[i]) return false;
		}
	}

	int* positions;
	int length;
};

class LeveledDAGNode {
public:
	LeveledDAGNode(int matchPointLength, int initValue) 
		: matchPoint(matchPointLength, initValue), incidentEdgeCount(0), partialLCSLength(0) {}
	~LeveledDAGNode() {
		for (int i = 0; i < releaseHeader.size(); i++) {
			delete[] releaseHeader[i];
		}
		delete[] matchPoint.positions;
	}

	void vectorRelease(std::vector<char*>& v) {
		for (int i = 0; i < v.size(); i++) {
			delete[] v[i];
		}
	}

	void GenerateSuccessor(int* successorTable, std::map<MatchPoint, LeveledDAGNode*>& DAG,
		std::vector<LeveledDAGNode*>& nextLevel, LeveledDAGNode* endNode) {
		for (int i = 0; i < BASES_COUNT; i++) {
			MatchPoint newMatchPoint(matchPoint, i, successorTable);
			if (newMatchPoint.isValid()) {
				auto it = DAG.find(newMatchPoint);
				if (it == DAG.end()) {
					LeveledDAGNode* newDAGNode = new LeveledDAGNode(matchPoint.length, 0);
					newDAGNode->matchPoint = newMatchPoint;
					char* LCS = new char[SEQUENCE_LENGTH] {0};
					switch (i) {
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
					successors.push_back(newDAGNode);
					nextLevel.push_back(newDAGNode);
				}
				else {
					successors.push_back(it->second);
					it->second->incidentEdgeCount++;
				}
			}
		}

		if (successors.size() == 0) {
			successors.push_back(endNode);
			endNode->incidentEdgeCount++;
		}
	}

	void PrintMLCS() {
		char* output = new char[partialLCSLength + 1] {0};
		for (int i = 0; i < partialLCS.size(); i++) {
			memcpy(output, partialLCS[i], sizeof(char) * partialLCSLength);
			std::cout << output << std::endl;
		}
		delete[] output;
	}

	int MLCSLength() {
		return partialLCSLength;
	}

	MatchPoint matchPoint;
	int incidentEdgeCount;
	int partialLCSLength;
	NitrogenousBases symbol;
	std::vector<LeveledDAGNode*> successors;
	std::vector<char*> partialLCS;
	std::vector<char*> releaseHeader;
};